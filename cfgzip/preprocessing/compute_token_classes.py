
from tqdm import tqdm
import multiprocessing
from contextlib import nullcontext
from typing import Dict, Set, Tuple, Optional, Callable, List, Generator
from cfgzip.preprocessing.utils import sdict_iter, sdict_add, memory_guard


class WorkerGlobals:
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


def init_globals(
        grammar_preterminals_rev: Dict[int, Set[int]],
        nt_map: Dict[int, Set[int]],
        transition_map: Dict[Tuple[int,  int], Set[Tuple[int, ...]]],
        stack_adjacency: Dict[int, Set[Optional[int]]],
        start_symbol: int
    ):
    global worker_globals
    worker_globals = WorkerGlobals(grammar_preterminals_rev, nt_map, transition_map, stack_adjacency, start_symbol)


def rec(
        t: Tuple[int, ...],
        in_stack: Tuple[int, ...],
        out_stack: Tuple[int, ...],
        prev_symbol: Optional[int],
        prev_stacks: Set[Tuple[Tuple[int, ...], Tuple[int, ...], int]],
        pos: int,
        allow_bt: bool  # stack backtrack disallowed for in_stack = (start_symbol,)
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    if (in_stack, out_stack, pos) not in prev_stacks:
        prev_stacks.add((in_stack, out_stack, pos))

        if len(t) == 0:
            yield in_stack, out_stack
        elif t[0] in worker_globals.grammar_preterminals_rev:
            if len(out_stack) == 0:  # stack backtrack (hit phrase boundary)
                if allow_bt:
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


def compute_stack_in_out(
        args: Tuple[Tuple[int, ...], int]
) -> Tuple[int, Optional[Set[Tuple[Tuple[int, ...], Tuple[int, ...]]]]]:
    tok, t_id = args

    if (preterms := worker_globals.grammar_preterminals_rev.get(tok[0])) is None:
        return t_id, None

    stack_in_out = set(rec(tok, (), (), None, set(), 0, True))

    if any(worker_globals.start_symbol in worker_globals.nt_map.get(pt, ()) for pt in preterms):
        stack_in_out.update(
            rec(tok, (worker_globals.start_symbol,), (worker_globals.start_symbol,), None, set(), 0, False)
        )

    return t_id, stack_in_out


def compute_stack_adj(grammar: Dict[int, Set[Tuple[int, ...]]], start_symbol: int) -> Dict[int, Set[Optional[int]]]:
    all_symbols, neighbors = set(grammar.keys()), {}

    for _, beta in sdict_iter(grammar):
        all_symbols.update(beta[1:])

        for b0, b1 in zip(beta[1:-1], beta[2:], strict=True):
            if b1 in neighbors.keys(): neighbors[b1].add(b0)
            else: neighbors.update({b1: {b0}})

    all_symbols.remove(start_symbol)
    stack_adj = {}

    def rec_bigrams(x, visited):
        visited.add(x)

        for prod in grammar[x]:
            if len(prod) == 1: yield x  # unary production x -> ...
            elif (x_next := prod[-1]) not in visited: yield from rec_bigrams(x_next, visited)

    for a in all_symbols:
        stack_adj.update({a: {None}})  # None for init backtrack
        for b in neighbors.get(a, ()): stack_adj[a].update(rec_bigrams(b, set()))

    stack_adj.update({start_symbol: set()})

    return stack_adj


@memory_guard(0.95, multiprocessing=True)
def compute_displacements(
        grammar: Dict[str, Set[Tuple[str, ...]]],
        grammar_preterminals: Dict[str, Set[int]],
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

    all_symbols = set(grammar.keys())
    all_symbols.update(x for _, beta in sdict_iter(grammar) for x in beta)

    start_symbol = 0
    symbol_map = {'S': start_symbol}
    symbol_map.update({k: i for i, k in enumerate(sorted(list(all_symbols - {'S'})), start=1)})

    # start_symbol = 'S'
    # symbol_map = {'S': start_symbol}
    # symbol_map.update({k: k for i, k in enumerate(sorted(list(all_symbols - {'S'})), start=1)})

    token_chars = {c for tok, _ in tokens for c in set(tok)}
    grammar_preterminals_rev = {}

    for k, c in sdict_iter(grammar_preterminals):
        if c in token_chars: sdict_add(grammar_preterminals_rev, c, symbol_map[k])

    grammar = {symbol_map[a]: {tuple(symbol_map[b] for b in beta) for beta in prods} for a, prods in grammar.items()}
    stack_adj = compute_stack_adj(grammar, start_symbol)

    for a, (b, *beta) in sdict_iter(grammar):
        sdict_add(transition_map, (b, a), tuple(beta))
        sdict_add(nt_map, b, a)

    tokens = sorted(tokens, key=lambda x: x[1])  # determinism
    tasks = []

    for tok, t_id in tokens:
        if not (ignore_tokens(tok, t_id) or t_id == eos_token_id):
            if skip_compute_tokens(tok, t_id):
                skip_classes.append(t_id)
            else:
                tasks.append((tok, t_id))

    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    elif num_workers == 1:
        init_globals(grammar_preterminals_rev, nt_map, transition_map, stack_adj, start_symbol)
        pool = nullcontext()
        results = map(compute_stack_in_out, tasks)
    else:
        pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=init_globals,
            initargs=(grammar_preterminals_rev, nt_map, transition_map, stack_adj, start_symbol),
        )
        chunksize = max(1, len(tasks) // (num_workers * 16))
        results = pool.imap_unordered(compute_stack_in_out, tasks, chunksize=chunksize)

    with pool:
        for t_id, stack_in_out in (tqdm(results, total=len(tasks)) if use_tqdm else results):
            if stack_in_out:
                displacement = tuple(sorted(seq_ids.setdefault(seq, len(seq_ids)) for seq in stack_in_out))
                token_classes.setdefault(displacement, []).append(t_id)
            else:
                invalid_tokens.append(t_id)


    token_classes_out = [[x] for x in skip_classes]
    token_classes_out.extend(token_classes.values())

    return token_classes_out, invalid_tokens


def compute_token_classes(
        grammar: Dict[str, Set[Tuple[str, ...]]],
        char_preterminals: Dict[str, Set[int]],
        tokens: List[Tuple[Tuple[int, ...], int]],
        eos_token_id: int,
        skip_compute_tokens: Optional[Callable[[Tuple[int, ...], int], bool]],
        ignore_tokens: Optional[Callable[[Tuple[int, ...], int], bool]],
        n_logits: Optional[int],
        use_tqdm: bool,
        num_workers: int
) -> Tuple[List[int], List[int], List[bytes]]:
    token_classes, invalid_tokens = compute_displacements(
        grammar, char_preterminals, tokens, eos_token_id, skip_compute_tokens, ignore_tokens, use_tqdm, num_workers,
        is_multiprocessing=(num_workers > 1)  # D13: memory_guard reads this kwarg; the decorator default (multiprocessing=True) is overridden here every call
    )
    token_classes.append([eos_token_id])

    n_logits = max(max(v for _, v in tokens) + 1, len(tokens)) if n_logits is None else n_logits
    token_classes_vec = [-1] * n_logits

    class_representatives = [[]] * len(token_classes)
    tokens_rev: Dict[int, Tuple[int, ...]] = {v: k for k, v in tokens}

    for i, c in enumerate(token_classes):
        min_len = float('inf')

        for x in c:
            token_classes_vec[x] = i  # token id x belongs to class num i

            if (len_x := len(x_tok := tokens_rev[x])) < min_len:
                class_representatives[i], min_len = x_tok, len_x

    return token_classes_vec, invalid_tokens, [bytes(c) for c in class_representatives]
