
from interegular import parse_pattern
from interegular.fsm import anything_else
from typing import Tuple, Dict, Set, Optional
from cfgzip.preprocessing.parse_gbnf import parse_gbnf
from cfgzip.preprocessing.utils import sdict_add, sdict_iter


# gets 4-byte utf-8 codepoint for each unicode char ID---pads codepoints <4 bytes with None for uniform len
def enc_codepoint(cp: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    cp_enc = tuple(chr(cp).encode('utf-8'))

    return cp_enc + ((None,) * (4 - len(cp_enc)))


# converts terminal of the form A -> regex to byte-level GNF CFG derived from NFA of regex with start symbol A and
# rules of the form q_i -> b (q_k)
def regex_to_gnf_cfg(label: str, regex: str) -> Tuple[Dict[str, Set[Tuple[str, ...]]], Dict[str, Set[int]]]:
    char_fsm = parse_pattern(regex).to_fsm()
    sym_to_chars, ae_sym = {}, None
    nfa_templates, template_lens = {}, {}

    for ch, sym in char_fsm.alphabet.items():  # flip char => sym (ID) mapping to sym => char
        if ch is anything_else: ae_sym = sym
        else: sym_to_chars.setdefault(sym, set()).add(ord(ch))

    # expand anything_else to all non-explicit codepoints
    if ae_sym is not None and any(ae_sym in sym_map for sym_map in char_fsm.map.values()):
        explicit = {ch for ch, _ in char_fsm.alphabet.items() if ch is not anything_else}
        sym_to_chars.update({ae_sym: {
            enc_codepoint(x) for x in range(0x110000)
            if (not 0xD800 <= x < 0xE000) and chr(x) not in explicit
        }})

    sym_to_chars = {k: v if k == ae_sym else {enc_codepoint(x) for x in v} for k, v in sym_to_chars.items()}

    # pre-compute byte-level NFA templates that recognize each character set
    for sym, charset in sym_to_chars.items():
        trie, trie_buckets, seq_hash = {}, {}, {}

        # build the explict trie---it will always have a depth of exactly 4 (None padding; see enc_codepoint def)
        for c in charset:
            trie_c = trie
            for i in range(2): trie_c = trie_c.setdefault(c[i], {})
            trie_c.setdefault(c[2], set()).add(c[3])

        def get_seq_hash(in_set):  # hashes each unique tuple to a unique integer ID
            if (in_tuple := tuple(sorted(in_set))) in seq_hash.keys():
                return seq_hash[in_tuple]

            tuple_hash = len(seq_hash.keys())
            seq_hash.update({in_tuple: tuple_hash})

            return tuple_hash

        def item_hash_set(dict_items):  # turns a set of (int, int) pairs (dict items) into a set of int IDs
            for it in dict_items:
                if it not in seq_hash.keys():
                    seq_hash.update({it: len(seq_hash.keys())})

            return {seq_hash[x] for x in dict_items}

        # hash the trie to compress NFA states
        for b0, v0 in trie.items():
            v0_buckets = {}

            for b1, v1 in v0.items():
                v1_buckets = {}

                # hash the -1-byte sets, bucket -2-byte sets based on -1-byte set hashes
                for b2, v2 in v1.items():
                    v2_hashed = get_seq_hash(v2)
                    v1[b2] = v2_hashed
                    v1_buckets.setdefault(v2_hashed, set()).add(b2)

                # hash the (-2-byte set, -1-byte set hash) pairs
                v1_hashed = item_hash_set({(get_seq_hash(v), k) for k, v in v1_buckets.items()})
                v1_hashed = get_seq_hash(v1_hashed)
                # bucket -3-byte sets based on -2/-1 pair hashes
                v0[b1] = v1_hashed
                v0_buckets.setdefault(v1_hashed, set()).add(b1)

            # hash the (-3-byte set, -2/-1 pair hash) pairs
            v0_hashed = item_hash_set({(get_seq_hash(v), k) for k, v in v0_buckets.items()})
            v0_hashed = get_seq_hash(v0_hashed)
            # bucket -4-byte sets based on -2/-1 pair hashes
            trie[b0] = v0_hashed
            trie_buckets.setdefault(v0_hashed, set()).add(b0)

        seq_hash_rev: list[tuple[int, ...]] = [None] * len(seq_hash)  # inverts the tuple hashing
        for k, i in seq_hash.items(): seq_hash_rev[i] = k

        nfa_templates.update({sym: {-1: {}}})
        sym_nfa_template = nfa_templates[sym]
        state_count = [-1]

        # recursively build NFA template from hashed trie representation
        def rec(chars, seq_id, prev_state, depth):
            if depth == 2:  # last two bytes (terminates recursion)
                if (chars_next := seq_hash_rev[seq_id]) == (None,):  # set of 3-byte sequences (None padding)
                    # add chars-labeled edge from prev_state to final state (-2)
                    sym_nfa_template.setdefault(prev_state, {}).setdefault(-2, set()).update(chars)
                else:  # set of 4-byte sequences (no padding in seq)
                    state_count[0] += 1   # create new state state_count[0]
                    # add chars-labeled edge from prev_state to state_count[0]
                    sym_nfa_template.setdefault(prev_state, {}).setdefault(state_count[0], set()).update(chars)
                    # add chars-labeled edge from state_count[0] to final state (-2)
                    sym_nfa_template.setdefault(state_count[0], {}).setdefault(-2, set()).update(chars_next)
            else:
                for pair in seq_hash_rev[seq_id]:  # seq ID is a hashed pair
                    # pair = (hashed set of chars, hashed seq ID)
                    chars_next, seq_id_next = seq_hash_rev[pair]
                    chars_next = seq_hash_rev[chars_next]  # un-hash char set

                    if chars_next == (None,):  # end of codepoint sequence (None padding)
                        # add chars-labeled edge from prev_state to final state (-2)
                        sym_nfa_template.setdefault(prev_state, {}).setdefault(-2, set()).update(chars)
                    else:
                        state_count[0] += 1  # create new state state_count[0]
                        # add chars-labeled edge from prev_state to state_count[0]
                        sym_nfa_template.setdefault(prev_state, {}).setdefault(state_count[0], set()).update(chars)
                        # continue adding edges from node state_count[0]
                        rec(chars_next, seq_id_next, state_count[0], depth + 1)

        for seq_id0, chars0 in trie_buckets.items():
            rec(chars0, seq_id0, -1, 0)

        template_lens.update({sym: state_count[0] + 1})

    nfa, start_loop = {}, False
    next_id = max(char_fsm.states, default=-1) + 1

    # build byte-level NFA from char-level NFA (char_fsm) + byte-level templates
    for src, sym_map in char_fsm.map.items():
        src_str = label if src == char_fsm.initial else f'{label}[q{src}]'

        for sym, trgt in sym_map.items():
            trgt_str = f'{label}[q{trgt}]'
            if trgt == char_fsm.initial: start_loop = True  # start symbol is target of an edge

            # merge in sym NFA template, fill in placeholder state IDs
            for k0, v0 in nfa_templates[sym].items():
                k0_str = src_str if k0 == -1 else f'{label}[q{k0 + next_id}]'

                for k1, v1 in v0.items():
                    k1_str = trgt_str if k1 == -2 else f'{label}[q{k1 + next_id}]'
                    nfa.setdefault(k0_str, {}).setdefault(k1_str, set()).update(v1)

            next_id += template_lens[sym]  # add number of states in template (template_lens[sym]) to running count

    accept_states = {f'{label}[q{x}]' for x in char_fsm.finals}
    cfg, preterminals = {}, {}

    # in case there are also rules {label} -> a A_1 ... A_n in the full CFG that this one will be merged into
    if start_loop: nfa.update({f'{label}[q{char_fsm.initial}]': nfa[label]})

    # now we build the GNF CFG: for each edge q_i -- C_k --> q_n in the NFA (states q_i/q_n, byte set C_k), we add...
    # 1. a production q_i
    for k0 in nfa.keys():
        for k1, v in nfa[k0].items():
            # create preterminal P[C_k]: represents set of bytes
            # so we don't have to duplicate the rules for each byte in C_K
            v_lbl = f'{label}[P{len(preterminals)}]'
            preterminals.update({v_lbl: v})

            if k1 in accept_states:
                sdict_add(cfg, k0, (v_lbl,))  # ... a production q_i -> P[C_k]  if q_n is an accept state

                # ... and a production q_i -> P[C_k] q_n  if q_n is an accept state
                # *but* there is also an edge q_n -> [any state]
                if k1 in nfa.keys():
                    sdict_add(cfg, k0, (v_lbl, k1))
            else:  # ... a production q_i -> P[C_k] q_n  otherwise
                sdict_add(cfg, k0, (v_lbl, k1))

    return cfg, preterminals


# exposed function of this file
# note (not sure where else to put this): the PT[t] thing is a trick---for each terminal t, we add a preterminal
# PT[t] with PT[t] -> t. this way, all non-unary productions are of the form X -> Y_1 ... Y_n, where all Y_1 ... Y_n
# are non-terminals. all Y_1's will get replaced during GNF-ification, Y_2 thru Y_n remain as start symbols for
# NFA sub-grammars
def parse_cfg_str(
        cfg_str: str,
        start_symbol: str = 'root'
) -> Tuple[Dict[str, Set[Tuple[str, ...]]], Dict[str, Set[Tuple[str, ...]]], Dict[str, Set[int]], Dict[str, str]]:
    terminals, cfg_dict = parse_gbnf(cfg_str)  # parse the GBNF grammar

    if start_symbol not in cfg_dict:  # TODO: more rigorous checks
        available = sorted(cfg_dict.keys())
        raise ValueError(
            f"start symbol {start_symbol!r} not found in grammar; "
            f"available non-terminals: {available}"
        )

    cfg_dict.update({'S': cfg_dict.pop(start_symbol)})  # replace start symbol with 'S'
    nfa_grammar, preterminals, cfg_out, terminal_labels = {}, {}, {}, {}

    for t_label, t_regex in terminals.items():
        cfg_out.update({f'PT[{t_label}]': {(t_label,)}})
        terminal_labels.update({f'PT[{t_label}]': t_label})

        label_grammar, label_preterminals = regex_to_gnf_cfg(t_label, t_regex)
        nfa_grammar.update(label_grammar)
        preterminals.update(label_preterminals)

    for a, beta in sdict_iter(cfg_dict):
        sdict_add(
            cfg_out, a, tuple(f'PT[{x}]' if x in terminal_labels else ('ε' if x == '""' else x) for x in beta)
        )

    return cfg_out, nfa_grammar, preterminals, terminal_labels
