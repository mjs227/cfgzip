
from typing import Dict, Set, Tuple, Iterable
from cfgzip.preprocessing.utils import sdict_iter, sdict_add, sdict_remove, memory_guard


def remove_epsilon_productions(cfg: Dict[str, Set[tuple[str, ...]]]) -> None:
    empty, e_prev = {'ε'}, set()
    cfg_next, cfg_prev = {}, {k: set(v) for k, v in cfg.items()}
    poss_empty = set()

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
                        poss_empty.add(lhs)

                if lhs_empty:
                    empty.add(lhs)

        cfg_prev = {k: set(v) for k, v in cfg_next.items()}
        cfg_next.clear()

    def rec(in_seq, n, out_seq):
        if n == len(in_seq):
            yield out_seq
        else:
            yield from rec(in_seq, n + 1, out_seq + (in_seq[n],))

            if in_seq[n] in poss_empty:
                yield from rec(in_seq, n + 1, out_seq)

    poss_empty = poss_empty - empty
    e_prev.clear()

    while not e_prev == poss_empty:
        e_prev = set(poss_empty)

        for lhs in sorted(list(cfg_prev.keys())):
            if any(all(x in poss_empty for x in rhs) for rhs in cfg_prev[lhs]):
                poss_empty.add(lhs)

    for lhs in sorted(list(cfg_prev.keys())):
        for rhs in sorted(list(cfg_prev[lhs]), key=str):
            for rhs_new in set(rec(rhs, 0, ())):
                if len(rhs_new) > 0: sdict_add(cfg_next, lhs, rhs_new)

    cfg.clear()
    cfg.update(cfg_next)


def remove_unary_rules(cfg: Dict[str, Set[tuple[str, ...]]], terminals: Iterable[str]) -> Dict[str, Set[str]]:
    to_remove, self_loops = ('', ''), set()
    terminal_map = {t: {t} for t in terminals}

    while to_remove:
        x, y = to_remove

        if (y,) in cfg.get(x, ()):
            sdict_remove(cfg, x, (y,))

            if y in cfg.keys():
                for rhs in cfg[y]: sdict_add(cfg, x, rhs)
            if y in terminal_map.keys():
                if x in terminal_map.keys():
                    terminal_map[x].update(terminal_map[y])
                else:
                    terminal_map.update({x: set(terminal_map[y])})

        to_remove = None
        self_loops.clear()
        cfg_keys = sorted(list(cfg.keys()))

        for lhs in cfg_keys:
            prods = sorted(list(cfg[lhs]), key=str)

            for rhs in prods:
                if len(rhs) == 1:
                    if rhs[0] == lhs:
                        self_loops.add(lhs)
                    elif to_remove is None:
                        to_remove = (lhs, rhs[0])

        for x in self_loops: sdict_remove(cfg, x, (x,))

    used_symbols = {v0 for _, v in sdict_iter(cfg) for v0 in v}
    used_symbols.add('S')

    for k in list(terminal_map.keys()):
        if k not in used_symbols: terminal_map.pop(k)

    return terminal_map


def remove_unreachable_symbols(cfg: Dict[str, Set[Tuple[str, ...]]]) -> None:
    reachable, r_prev = {'S'}, set()

    while not r_prev == reachable:
        r_update = sorted(list(reachable - r_prev))
        r_prev = set(reachable)
        reachable.update(x for r in r_update for r_prod in sorted(list(cfg.get(r, [])), key=str) for x in r_prod)

    for k in cfg.keys() - reachable: cfg.pop(k)


def cfg_to_gnf(cfg: Dict[str, Set[Tuple[str, ...]]], preterminals: Set[str]):
    def loop(cfg_key_iter):
        for key in cfg_key_iter:
            key_prods, key_prods_new = set(cfg[key]), set()

            while not key_prods_new == key_prods:
                for key_beta in sorted(list(key_prods)):
                    yield key, key_beta, key_prods_new, key_prods

                if key_prods_new == key_prods: break
                key_prods = set(key_prods_new)
                key_prods_new.clear()

            cfg[key] = key_prods

    for k, beta, k_prods_new, k_prods in loop(list(cfg.keys())):  # remove local left recursions
        if beta[0] == k:
            sdict_add(cfg, f'LR[{k}]', beta[1:])
            sdict_add(cfg, f'LR[{k}]', beta[1:] + (f'LR[{k}]',))

            for prod in k_prods:
                if not (prod[0] == k or prod[-1] == f'LR[{k}]'):
                    k_prods_new.add(prod + (f'LR[{k}]',))
        else:
            k_prods_new.add(beta)

    cfg_keys, prev_level, visited = ['S'], ['S'], {'S'}

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

    for k, beta, k_prods_new, k_prods in loop(cfg_keys[2:]):  # remove global (indirect) left-recursions
        if (beta0 := beta[0]) in preterminals or cfg_key_ords[beta0] > cfg_key_ords[k]:
            k_prods_new.add(beta)
        elif beta0 == k:  # local recursion surfacing
            sdict_add(cfg, f'LR[{k}]', beta[1:])
            sdict_add(cfg, f'LR[{k}]', beta[1:] + (f'LR[{k}]',))

            for prod in k_prods:
                if not (prod[0] == k or prod[-1] == f'LR[{k}]'):
                    k_prods_new.add(prod + (f'LR[{k}]',))
        else:
            beta1 = beta[1:]
            for prod in cfg[beta0]: k_prods_new.add(prod + beta1)

    cfg_keys += [k for k in cfg.keys() if k not in keys_before]  # GNF-ify must reach the new LR[k] keys

    for k, beta, k_prods_new, _ in loop(cfg_keys):  # GNF-ify
        if (beta0 := beta[0]) in preterminals:
            k_prods_new.add(beta)
        else:
            beta1 = beta[1:]
            for prod in cfg[beta0]: k_prods_new.add(prod + beta1)

    reachable, reachable_prev = {'S'}, {'S'}

    while True:
        for a, beta in sdict_iter(cfg):
            if a in reachable_prev:
                for b in beta[1:]: reachable.add(b)

        if reachable == reachable_prev: break
        reachable_prev.update(reachable)

    for k in cfg.keys() - reachable: cfg.pop(k)

    while True:
        # never merge the start symbol: dropping it loses the entry rule, and folding another
        # (self-recursive) symbol into it would put 'S' on a RHS, which the downstream stack
        # adjacency in compute_token_classes assumes never happens
        cfg_keys = list(cfg.keys() - {'S'})

        for i, k0 in enumerate(cfg_keys[:-1], start=1):
            for k1 in cfg_keys[i:]:
                if cfg[k0] == cfg[k1]:
                    for k2 in cfg.keys():
                        cfg[k2] = {tuple(k0 if b == k1 else b for b in beta) for beta in cfg[k2]}

                    cfg.pop(k1)
                    break
            else:
                continue
            break
        else:
            break

    return cfg


@memory_guard(0.95)
def normalize_cfg(
        cfg: Dict[str, Set[Tuple[str, ...]]],
        nfa_grammar: Dict[str, Set[Tuple[str, ...]]],
        terminal_labels: Set[str]
) -> Dict[str, Set[Tuple[str, str]]]:
    cfg_terminals = {f'PT[{t}]': t for t in terminal_labels}
    remove_epsilon_productions(cfg)
    terminal_map = remove_unary_rules(cfg, cfg_terminals.keys())
    remove_unreachable_symbols(cfg)

    for x in sorted(list(terminal_map.keys())):
        for y in sorted(list(terminal_map[x])):
            sdict_add(cfg, x, (cfg_terminals[y],))

    cfg = cfg_to_gnf(cfg, set(cfg_terminals.values()))
    cfg_out = {}

    for a, beta in sdict_iter(cfg):
        if a not in cfg_terminals.keys():  # temporary, get removed
            beta1 = tuple(cfg_terminals.get(x, x) for x in beta[1:])  # removing "PT[...]"

            for nfa_prod in nfa_grammar[beta[0]]:
                sdict_add(cfg_out, a, nfa_prod + beta1)

    all_symbols = {x for _, beta in sdict_iter(cfg_out) for x in beta}
    cfg_out.update({k: v for k, v in nfa_grammar.items() if '[q' in k or k in all_symbols})

    return cfg_out
