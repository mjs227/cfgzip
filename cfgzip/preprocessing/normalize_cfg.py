
from typing import Dict, Set, Tuple, Iterable
from cfgzip.preprocessing.utils import sdict_iter, sdict_add, sdict_remove, memory_guard


def remove_epsilon_productions(cfg: Dict[str, Set[tuple[str, ...]]]) -> None:
    empty, e_prev = {'ε'}, set()  # empty: symbols that derive ONLY ε
    cfg_next, cfg_prev = {}, {k: set(v) for k, v in cfg.items()}
    poss_empty = set()  # can derive ε AND other things too

    # strip symbols in `empty` out of every RHS; symbols A s.t. every RHS of A vanishes join `empty`;
    # loop until convergence
    while not e_prev == empty:
        e_prev = set(empty)

        for lhs in sorted(list(cfg_prev.keys())):
            if lhs not in empty:
                lhs_empty = True

                for rhs in sorted(list(cfg_prev[lhs]), key=str):
                    rhs_new = tuple(x for x in rhs if x not in empty)

                    if len(rhs_new) > 0:
                        sdict_add(cfg_next, lhs, rhs_new)
                        lhs_empty = False
                    else:
                        poss_empty.add(lhs)  # rhs fully vanished => lhs can derive ε

                if lhs_empty:
                    empty.add(lhs)  # every RHS vanished => lhs derives only ε

        cfg_prev = {k: set(v) for k, v in cfg_next.items()}
        cfg_next.clear()

    # enumerates every subset of poss_empty symbols from a production (in_seq),
    # and yields each variant of in_seq with a subset dropped
    def rec(in_seq, n, out_seq):
        if n == len(in_seq):
            yield out_seq
        else:
            yield from rec(in_seq, n + 1, out_seq + (in_seq[n],))  # keep symbol n

            if in_seq[n] in poss_empty:
                yield from rec(in_seq, n + 1, out_seq)  # ... and drop it, if poss_empty

    poss_empty = poss_empty - empty  # fully-empty symbols are already gone from the RHSs above
    e_prev.clear()

    # close poss_empty under [some RHS can be entirely empty] => [its LHS is poss_empty too]
    while not e_prev == poss_empty:
        e_prev = set(poss_empty)

        for lhs in sorted(list(cfg_prev.keys())):
            if any(all(x in poss_empty for x in rhs) for rhs in cfg_prev[lhs]):
                poss_empty.add(lhs)

    # add each production in every variant with poss_empty symbols optionally removed
    for lhs in sorted(list(cfg_prev.keys())):
        for rhs in sorted(list(cfg_prev[lhs]), key=str):
            for rhs_new in set(rec(rhs, 0, ())):
                if len(rhs_new) > 0: sdict_add(cfg_next, lhs, rhs_new)

    cfg.clear()
    cfg.update(cfg_next)  # update the CFG in-place


# removes unary productions (i.e. A -> B for single non-terminal B) in-place;
# returns terminal_map: symbol -> transitive closure (via unit-rule chains) of (pre)terminals it can produce
# (keys are non-terminals plus the PT[t] preterminals themselves);
# `terminals` arg = the PT[t] keys
def remove_unary_rules(cfg: Dict[str, Set[tuple[str, ...]]], terminals: Iterable[str]) -> Dict[str, Set[str]]:
    to_remove, self_loops = ('', ''), set()  # self_loops: set of symbols A with A -> A
    terminal_map = {t: {t} for t in terminals}

    while to_remove:  # process one pending unit rule (x -> y) at a time until none are left
        x, y = to_remove

        if (y,) in cfg.get(x, ()):
            sdict_remove(cfg, x, (y,))

            if y in cfg.keys():
                for rhs in cfg[y]: sdict_add(cfg, x, rhs)  # give all of y's productions to x
            if y in terminal_map.keys():  # x also inherits whichever terminals y produces (if applicable)
                if x in terminal_map.keys(): terminal_map[x].update(terminal_map[y])
                else: terminal_map.update({x: set(terminal_map[y])})

        to_remove = None  # loop breaks if no new to_remove is found
        self_loops.clear()
        cfg_keys = sorted(list(cfg.keys()))  # sorting for determinism (for debugging)

        for lhs in cfg_keys:  # rescan for the next unit rule + collect self-loops (A -> A)
            prods = sorted(list(cfg[lhs]), key=str)

            for rhs in prods:
                if len(rhs) == 1:
                    if rhs[0] == lhs: self_loops.add(lhs)
                    elif to_remove is None: to_remove = (lhs, rhs[0])

        for x in self_loops: sdict_remove(cfg, x, (x,))  # A -> A is vacuous, just delete it

    used_symbols = {v0 for _, v in sdict_iter(cfg) for v0 in v}
    used_symbols.add('S')

    for k in list(terminal_map.keys()):  # remove unused (unreachable) symbols from terminal_map
        if k not in used_symbols: terminal_map.pop(k)

    return terminal_map


# drops symbols unreachable from the start symbol; in place; (this is just a BFS)
def remove_unreachable_symbols(cfg: Dict[str, Set[Tuple[str, ...]]]) -> None:
    reachable, r_prev = {'S'}, set()

    while not r_prev == reachable:
        r_update = sorted(reachable - r_prev)
        r_prev = set(reachable)
        reachable.update(x for r in r_update for r_prod in sorted(cfg.get(r, ()), key=str) for x in r_prod)

    for k in cfg.keys() - reachable: cfg.pop(k)


# converts the CFG to Greibach Normal Form (GNF) via Paull's algorithm
def cfg_to_gnf(cfg: Dict[str, Set[Tuple[str, ...]]], preterminals: Set[str]):
    # helper to limit code re-use: repeatedly sweep productions of `key` (so the caller can rewrite each prod
    # into new_prods) until convergence
    def loop(cfg_key_iter):
        for key in cfg_key_iter:
            key_prods, key_prods_new = set(cfg[key]), set()

            while not key_prods_new == key_prods:
                for key_beta in sorted(list(key_prods)):
                    yield key, key_beta, key_prods_new, key_prods

                if key_prods_new == key_prods: break
                key_prods = set(key_prods_new)
                key_prods_new.clear()

            cfg[key] = key_prods  # update cfg[key] upon convergence

    for k, beta, k_prods_new, k_prods in loop(list(cfg.keys())):  # remove local left recursions (A -> A ...)
        if beta[0] == k:
            sdict_add(cfg, f'LR[{k}]', beta[1:])
            sdict_add(cfg, f'LR[{k}]', beta[1:] + (f'LR[{k}]',))

            for prod in k_prods:
                if not (prod[0] == k or prod[-1] == f'LR[{k}]'):
                    k_prods_new.add(prod + (f'LR[{k}]',))
        else:
            k_prods_new.add(beta)

    cfg_keys, prev_level, visited = ['S'], ['S'], {'S'}

    # order the non-terminals by BFS from S (Paull's algorithm requires fixed ordering); within a
    # level, sort by total RHS length (heuristic: substituting big symbols later means fewer substitutions)
    while prev_level:
        next_level = []

        for x in prev_level:
            for beta in cfg[x]:
                for b in beta:
                    if b in cfg.keys() and b not in visited:
                        next_level.append(b)
                        visited.add(b)

        next_level.sort(key=(lambda z: -sum(len(w) for w in cfg[z])))
        cfg_keys.extend(next_level)
        prev_level = next_level

    cfg_key_ords = {k: i for i, k in enumerate(cfg_keys)}
    keys_before = set(cfg.keys())

    # remove global (indirect) left-recursions; safe to skip S (idx 0) and idx 1 (`cfg_keys[:2]`): idx 1's only
    # predecessor is S, which never appears on an RHS
    for k, beta, k_prods_new, k_prods in loop(cfg_keys[2:]):
        if (beta0 := beta[0]) in preterminals or cfg_key_ords[beta0] > cfg_key_ords[k]:
            k_prods_new.add(beta)  # already starts with (pre)terminal, or ordered after k: nothing to do
        elif beta0 == k:  # substitution re-surfaced a direct self-loop -> same LR[k] rewrite as the local loop above
            sdict_add(cfg, f'LR[{k}]', beta[1:])
            sdict_add(cfg, f'LR[{k}]', beta[1:] + (f'LR[{k}]',))

            for prod in k_prods:
                if not (prod[0] == k or prod[-1] == f'LR[{k}]'):
                    k_prods_new.add(prod + (f'LR[{k}]',))
        else:  # beta0 ordered before k: A -> B γ => {A -> δ γ | B -> δ}
            beta1 = beta[1:]
            for prod in cfg[beta0]: k_prods_new.add(prod + beta1)

    cfg_keys += [k for k in cfg.keys() if k not in keys_before]  # add the new LR[k] keys

    # CFG is now left-recursion-free, so we can GNF-ify: replace each leading non-terminal beta0 with
    # each of its productions until convergence (i.e. GNF)
    for k, beta, k_prods_new, _ in loop(cfg_keys):
        if (beta0 := beta[0]) in preterminals:
            k_prods_new.add(beta)
        else:
            beta1 = beta[1:]
            for prod in cfg[beta0]: k_prods_new.add(prod + beta1)

    # GNF substitution can orphan symbols
    remove_unreachable_symbols(cfg)

    # deduplication: merge non-terminals with identical production sets until convergence
    while True:
        # never merge the start symbol: dropping it loses the root, and folding another
        # (self-recursive) symbol into it would put 'S' on an RHS, which the downstream stack
        # adjacency in compute_token_classes assumes never happens
        cfg_keys = list(cfg.keys() - {'S'})

        for i, k0 in enumerate(cfg_keys[:-1], start=1):
            for k1 in cfg_keys[i:]:
                if cfg[k0] == cfg[k1]:
                    for k2 in cfg.keys():  # rewrite every reference to k1 over to k0
                        cfg[k2] = {tuple(k0 if b == k1 else b for b in beta) for beta in cfg[k2]}

                    cfg.pop(k1)
                    break
            else:
                continue
            break
        else:
            break

    return cfg


# exposed function of this file:
# normalizes a CFG to Greibach Normal Form (GNF) so each rule A -> a β maps onto a PDA transition
@memory_guard(0.95)
def normalize_cfg(
        cfg: Dict[str, Set[Tuple[str, ...]]],  # non-terminal -> set of productions
        nfa_grammar: Dict[str, Set[Tuple[str, ...]]],  # terminal name -> resp. byte-level NFA grammar
        terminal_labels: Dict[str, str]  # maps PT[t] -> t (see parse_cfg_str.py)
) -> Dict[str, Set[Tuple[str, str]]]:
    remove_epsilon_productions(cfg)
    terminal_map = remove_unary_rules(cfg, terminal_labels.keys())
    remove_unreachable_symbols(cfg)

    # introduce the productions x -> PT[t] (see parse_cfg_str.py) folded into terminal_map in remove_unary_rules()
    for x in sorted(list(terminal_map.keys())):
        for y in sorted(list(terminal_map[x])):
            sdict_add(cfg, x, (terminal_labels[y],))

    cfg = cfg_to_gnf(cfg, set(terminal_labels.values()))
    cfg_out = {}

    # splice the NFA into the CFG: expand each leading (pre)terminal label into its byte-level NFA init productions
    for a, beta in sdict_iter(cfg):
        if a not in terminal_labels.keys():  # temporary PT[t] symbols w/ only unary prods PT[t] -> t
            beta1 = tuple(terminal_labels.get(x, x) for x in beta[1:])  # remove any `PT[...]` left in the tail

            for nfa_prod in nfa_grammar[beta[0]]:  # leading PT[t] got replaced w/ t in cfg_to_gnf()
                sdict_add(cfg_out, a, nfa_prod + beta1)

    # add in NFA internal state rules (keys containing '[q') + any other referenced NFA symbols
    all_symbols = {x for _, beta in sdict_iter(cfg_out) for x in beta}
    cfg_out.update({k: v for k, v in nfa_grammar.items() if '[q' in k or k in all_symbols})

    return cfg_out
