
from tqdm import tqdm
import multiprocessing
from contextlib import nullcontext
from typing import Dict, Set, Tuple, Optional, Callable, List, Generator
from cfgzip.preprocessing.utils import sdict_iter, sdict_add, memory_guard


class WorkerGlobals:  # stores global read-only objects for worker threads
    __slots__ = ('grammar_preterminals_rev', 'nt_map', 'transition_map', 'stack_adjacency', 'start_symbol')

    def __init__(
            self,
            grammar_preterminals_rev: Dict[int, Set[int]],
            nt_map: Dict[int, Set[int]],
            transition_map: Dict[Tuple[int,  int], Set[Tuple[int, ...]]],
            stack_adjacency: Dict[int, Set[Optional[int]]],
            start_symbol: int
    ):
        self.grammar_preterminals_rev, self.nt_map = grammar_preterminals_rev, nt_map
        self.transition_map, self.stack_adjacency, self.start_symbol = transition_map, stack_adjacency, start_symbol


worker_globals: WorkerGlobals = None


def init_globals(  # initializes worker_globals on a given thread
        grammar_preterminals_rev: Dict[int, Set[int]],
        nt_map: Dict[int, Set[int]],
        transition_map: Dict[Tuple[int,  int], Set[Tuple[int, ...]]],
        stack_adjacency: Dict[int, Set[Optional[int]]],
        start_symbol: int
    ):
    global worker_globals
    worker_globals = WorkerGlobals(grammar_preterminals_rev, nt_map, transition_map, stack_adjacency, start_symbol)


def rec(  # implements algorithm 1 from the paper
        t: Tuple[int, ...],  # input token
        in_stack: Tuple[int, ...],  # current input stack
        out_stack: Tuple[int, ...],  # current output stack
        prev_symbol: Optional[int],  # prev. symbol (for stack-adj restriction): initializes to None)
        prev_stacks: Set[Tuple[Tuple[int, ...], Tuple[int, ...], int]],  # cache prev stacks for dedup
        pos: int,  # current token byte pos (= recursion depth)
        allow_bt: bool  # stack backtrack disallowed for in_stack = (start_symbol,)
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    # no need to re-compute if we've already hit this state at this pos
    if (in_stack, out_stack, pos) not in prev_stacks:
        prev_stacks.add((in_stack, out_stack, pos))  # cache state + pos

        if len(t) == 0:  # end-of-token
            yield in_stack, out_stack
        elif t[0] in worker_globals.grammar_preterminals_rev:  # "char in vocab"
            if len(out_stack) == 0:  # stack backtrack (hit phrase boundary)
                if allow_bt:  # stack backtrack disallowed for initial in_stack = (start_symbol,)
                    for t0_pt in worker_globals.grammar_preterminals_rev[t[0]]:
                        if t0_pt in worker_globals.nt_map:
                            for s in worker_globals.nt_map[t0_pt]:
                                if prev_symbol in worker_globals.stack_adjacency[s]:
                                    for t0_out in worker_globals.transition_map[(t0_pt, s)]:
                                        yield from rec(
                                            t[1:], in_stack + (s,), t0_out, s, prev_stacks, pos + 1, allow_bt
                                        )
            else:
                for t0_pt in worker_globals.grammar_preterminals_rev[t[0]]:
                    if (t0_pt, out_stack[0]) in worker_globals.transition_map:
                        for t0_out in worker_globals.transition_map[(t0_pt, out_stack[0])]:
                            yield from rec(
                                t[1:], in_stack, t0_out + out_stack[1:], out_stack[0], prev_stacks, pos + 1, allow_bt
                            )


# basically just wraps rec() above; returns token ID + {all (in_stack, out_stack) pairs}
def compute_stack_in_out(
        args: Tuple[Tuple[int, ...], int]  # args packaged as tuple b/c of multiprocessing
) -> Tuple[int, Optional[Set[Tuple[Tuple[int, ...], Tuple[int, ...]]]]]:
    tok, t_id = args

    # no preterminals generate the initial byte of the token => never valid
    if (preterms := worker_globals.grammar_preterminals_rev.get(tok[0])) is None:
        return t_id, None

    # compute (in, out) pairs for all *non-start* symbols:
    # stack_adj rel. is def'd to prevent start from appearing in initial backtrack
    stack_in_out = set(rec(tok, (), (), None, set(), 0, True))

    # compute (in, out) pairs starting w/ start symbol: no backtrack allowed---this is the case where the token
    # is the very first generated token (=> can't backtrack)
    if any(worker_globals.start_symbol in worker_globals.nt_map.get(pt, ()) for pt in preterms):
        stack_in_out.update(
            rec(tok, (worker_globals.start_symbol,), (worker_globals.start_symbol,), None, set(), 0, False)
        )

    return t_id, stack_in_out


# pre-computes the stack adjacency relation
def compute_stack_adj(grammar: Dict[int, Set[Tuple[int, ...]]], start_symbol: int) -> Dict[int, Set[Optional[int]]]:
    all_symbols, neighbors = set(grammar.keys()), {}

    # neighbors[Y] = set of all X s.t. exists rule A -> ... X Y ...
    for _, beta in sdict_iter(grammar):
        all_symbols.update(beta[1:])

        for b0, b1 in zip(beta[1:-1], beta[2:], strict=True):
            if b1 in neighbors.keys(): neighbors[b1].add(b0)
            else: neighbors.update({b1: {b0}})

    # remove start symbol to prevent it appearing in initial backtrack
    all_symbols.remove(start_symbol)
    stack_adj = {}

    # recursively computes stack adj (duh!): see Appendix C in the paper
    def rec_adj(x, visited):
        visited.add(x)

        for prod in grammar[x]:
            if len(prod) == 1: yield x  # unary production x -> ...
            elif (x_next := prod[-1]) not in visited: yield from rec_adj(x_next, visited)

    for a in all_symbols:
        stack_adj.update({a: {None}})  # None for init backtrack
        for b in neighbors.get(a, ()): stack_adj[a].update(rec_adj(b, set()))

    stack_adj.update({start_symbol: set()})

    return stack_adj


# compute displacements for all tokens in vocab, then sort into classes
@memory_guard(0.95, multiprocessing=True)  # prevents crash (this fn causes lots of OOM)
def precomp_token_classes(  # see compute_token_classes() for param descriptions
        grammar: Dict[str, Set[Tuple[str, ...]]],
        preterminals: Dict[str, Set[int]],
        tokens: List[Tuple[Tuple[int, ...], int]],
        eos_token_id: int,
        skip_compute_tokens: Optional[Callable[[Tuple[int, ...], int], bool]],
        ignore_tokens: Optional[Callable[[Tuple[int, ...], int], bool]],
        use_tqdm: bool,
        num_workers: int
) -> Tuple[List[List[int]], List[int]]:
    skip_compute_tokens = (lambda *_: False) if skip_compute_tokens is None else skip_compute_tokens
    ignore_tokens = (lambda *_: False) if ignore_tokens is None else ignore_tokens
    nt_map, transition_map, seq_ids, token_classes, skip_classes, invalid_tokens = {}, {}, {}, {}, [], []

    # map str symbols to int IDs (more memory-efficient)
    all_symbols = set(grammar.keys())
    all_symbols.update(x for _, beta in sdict_iter(grammar) for x in beta)
    start_symbol = 0
    symbol_map = {'S': start_symbol}
    symbol_map.update({k: i for i, k in enumerate(sorted(list(all_symbols - {'S'})), start=1)})

    # replace str symbols with int IDs
    grammar = {symbol_map[a]: {tuple(symbol_map[b] for b in beta) for beta in prods} for a, prods in grammar.items()}
    stack_adj = compute_stack_adj(grammar, start_symbol)

    token_chars = {c for tok, _ in tokens for c in set(tok)}  # all possible byte IDs in the token vocab
    preterminals_rev = {}  # maps byte ID i to set of all preterminals P s.t. P -> i

    for k, c in sdict_iter(preterminals):
        if c in token_chars: sdict_add(preterminals_rev, c, symbol_map[k])

    # transition map = PDA transition fn
    # nt_map: maps preterminals P to set of all non-terminals A s.t. A -> P ...
    for a, (b, *beta) in sdict_iter(grammar):
        sdict_add(transition_map, (b, a), tuple(beta))
        sdict_add(nt_map, b, a)

    tokens = sorted(tokens, key=lambda x: x[1])  # inject determinism for e.g. debugging
    tasks = []  # multiprocessing jobs

    for tok, t_id in tokens:
        if not (ignore_tokens(tok, t_id) or t_id == eos_token_id):
            if skip_compute_tokens(tok, t_id): skip_classes.append(t_id)
            else: tasks.append((tok, t_id))

    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    elif num_workers == 1:  # run single-threaded (duh!)
        init_globals(preterminals_rev, nt_map, transition_map, stack_adj, start_symbol)
        pool = nullcontext()
        results = map(compute_stack_in_out, tasks)
    else:
        pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=init_globals,
            initargs=(preterminals_rev, nt_map, transition_map, stack_adj, start_symbol),
        )
        chunksize = max(1, len(tasks) // (num_workers * 16))  # size of blocks of jobs passed to each worker
        results = pool.imap_unordered(compute_stack_in_out, tasks, chunksize=chunksize)

    # get (ID(t), disp(t)) pairs, bucket tokens into equiv. classes by displacement
    with pool:
        for t_id, stack_in_out in (tqdm(results, total=len(tasks)) if use_tqdm else results):
            if stack_in_out:
                displacement = tuple(sorted(seq_ids.setdefault(seq, len(seq_ids)) for seq in stack_in_out))
                token_classes.setdefault(displacement, []).append(t_id)
            else:
                invalid_tokens.append(t_id)  # empty displacement => not valid in any context

    token_classes_out = [[x] for x in skip_classes]  # skipped tokens (too expensive) get their own singleton classes
    token_classes_out.extend(token_classes.values())

    return token_classes_out, invalid_tokens


# exposed function of this file
def compute_token_classes(
        grammar: Dict[str, Set[Tuple[str, ...]]],  # dict: maps non-terminal to set of all corresp. productions
        preterminals: Dict[str, Set[int]],  # maps preterm P to set of all byte IDs i s.t. P -> i
        tokens: List[Tuple[Tuple[int, ...], int]],  # pairs: (tuple of byte IDs, token ID)
        eos_token_id: int,
        skip_compute_tokens: Optional[Callable[[Tuple[int, ...], int], bool]],  # tokens that are too expensive
        ignore_tokens: Optional[Callable[[Tuple[int, ...], int], bool]],  # tokens that should be ignored (special)
        n_logits: Optional[int],  # num logits: to be used when tokenizer doesn't explicitly give number of logits
        use_tqdm: bool,
        num_workers: int  # for multiprocessing
) -> Tuple[List[int], List[int], List[bytes]]:
    # compute raw classes with precomp_token_classes
    # D13 (is_multiprocessing): memory_guard reads this kwarg;
    # the decorator default (multiprocessing=True) is overridden here every call
    token_classes, invalid_tokens = precomp_token_classes(
        grammar, preterminals, tokens, eos_token_id, skip_compute_tokens, ignore_tokens, use_tqdm, num_workers,
        is_multiprocessing=(num_workers > 1)
    )
    token_classes.append([eos_token_id])  # eos token gets its own ID

    n_logits = max(max(v for _, v in tokens) + 1, len(tokens)) if n_logits is None else n_logits
    token_classes_vec = [-1] * n_logits

    class_representatives = [[]] * len(token_classes)
    tokens_rev: Dict[int, Tuple[int, ...]] = {v: k for k, v in tokens}

    # convert list of lists of token IDs (tokens sorted into classes) into vocab-len list of class IDs
    for i, c in enumerate(token_classes):
        min_len = float('inf')

        for x in c:
            token_classes_vec[x] = i  # token id x belongs to class num i

            # class representative is byte-wise shortest token
            if (len_x := len(x_tok := tokens_rev[x])) < min_len:
                class_representatives[i], min_len = x_tok, len_x

    return token_classes_vec, invalid_tokens, [bytes(c) for c in class_representatives]
