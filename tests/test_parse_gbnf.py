import re
import pytest
from cfgzip.preprocessing.parse_gbnf import parse_gbnf, gbnf_to_regex


def _matches(regex, s):
    return bool(re.compile(regex).fullmatch(s))


# ──────────────────────────────────────────────
# gbnf_to_regex unit tests
# ──────────────────────────────────────────────

def test_gbnf_to_regex_string_simple():
    assert gbnf_to_regex('"abc"') == "abc"


def test_gbnf_to_regex_string_special_chars():
    r = gbnf_to_regex('"a.b"')
    assert _matches(r, "a.b")
    assert not _matches(r, "axb")


def test_gbnf_to_regex_char_class():
    r = gbnf_to_regex('[a-z]')
    assert _matches(r, "a") and _matches(r, "z")
    assert not _matches(r, "A") and not _matches(r, "0")


def test_gbnf_to_regex_char_class_negated():
    r = gbnf_to_regex(r'[^\n]')
    assert _matches(r, "a") and _matches(r, " ")
    assert not _matches(r, "\n")


def test_gbnf_to_regex_string_plus():
    r = gbnf_to_regex('"abc"+')
    assert _matches(r, "abc") and _matches(r, "abcabc")
    assert not _matches(r, "ab") and not _matches(r, "")


def test_gbnf_to_regex_charclass_plus():
    r = gbnf_to_regex('[a-z]+')
    assert _matches(r, "a") and _matches(r, "abc")
    assert not _matches(r, "") and not _matches(r, "A")


def test_gbnf_to_regex_charclass_star():
    r = gbnf_to_regex('[a-z]*')
    assert _matches(r, "") and _matches(r, "a") and _matches(r, "abc")
    assert not _matches(r, "A")


def test_gbnf_to_regex_charclass_optional():
    r = gbnf_to_regex('[0-9]?')
    assert _matches(r, "") and _matches(r, "5")
    assert not _matches(r, "55")


def test_gbnf_to_regex_concat_string_charclass():
    r = gbnf_to_regex('"ab" [0-9]')
    assert _matches(r, "ab5")
    assert not _matches(r, "ab") and not _matches(r, "5")


def test_gbnf_to_regex_alternation():
    r = gbnf_to_regex('("a" | "b")')
    assert _matches(r, "a") and _matches(r, "b")
    assert not _matches(r, "c") and not _matches(r, "ab")


def test_gbnf_to_regex_group_plus():
    r = gbnf_to_regex('("a" | "b")+')
    assert _matches(r, "a") and _matches(r, "b") and _matches(r, "ab") and _matches(r, "aab")
    assert not _matches(r, "") and not _matches(r, "c")


def test_gbnf_to_regex_nested():
    r = gbnf_to_regex('([a-z] [0-9]*)')
    assert _matches(r, "a") and _matches(r, "a1") and _matches(r, "a123")
    assert not _matches(r, "1") and not _matches(r, "")


def test_gbnf_to_regex_escape_hex():
    r = gbnf_to_regex(r'"\x41"')  # \x41 = 'A'
    assert _matches(r, "A")
    assert not _matches(r, "a")


# ──────────────────────────────────────────────
# parse_gbnf: terminal extraction
# ──────────────────────────────────────────────

def test_terminal_extraction_simple():
    terminals, prods = parse_gbnf('root ::= "hello"')
    assert len(terminals) == 1
    t_label = list(terminals.keys())[0]
    assert _matches(terminals[t_label], "hello")
    assert not _matches(terminals[t_label], "hell")
    assert not _matches(terminals[t_label], "helloo")
    assert prods["root"] == {(t_label,)}


def test_terminal_extraction_reuse():
    terminals, prods = parse_gbnf('root ::= "a" "a"')
    assert len(terminals) == 1
    t = list(terminals.keys())[0]
    assert prods["root"] == {(t,)}


def test_terminal_extraction_char_class():
    terminals, prods = parse_gbnf('root ::= [a-z]+')
    assert len(terminals) == 1
    t = list(terminals.keys())[0]
    assert _matches(terminals[t], "a") and _matches(terminals[t], "abc")
    assert not _matches(terminals[t], "") and not _matches(terminals[t], "A")
    assert prods["root"] == {(t,)}


def test_terminal_mixed():
    terminals, prods = parse_gbnf('root ::= "id_" [a-z]+')
    assert len(terminals) == 1, f"expected 1 terminal, got {len(terminals)}: {terminals}"
    t = list(terminals.keys())[0]
    assert _matches(terminals[t], "id_a") and _matches(terminals[t], "id_abc")
    assert not _matches(terminals[t], "id_") and not _matches(terminals[t], "_a")
    assert prods["root"] == {(t,)}


# ──────────────────────────────────────────────
# parse_gbnf: alternation (the | operator)
# ──────────────────────────────────────────────

def test_alternation_basic():
    terminals, prods = parse_gbnf('root ::= "a" | "b"')
    assert len(terminals) == 2
    t_a = next(k for k, v in terminals.items() if re.fullmatch(v, "a"))
    t_b = next(k for k, v in terminals.items() if re.fullmatch(v, "b"))
    assert prods["root"] == {(t_a,), (t_b,)}


def test_alternation_multi_symbol():
    terminals, prods = parse_gbnf('root ::= "a" "b" | "c"')
    sizes = sorted(len(tup) for tup in prods["root"])
    assert sizes == [1, 1], f"expected [1, 1], got {sizes}"


# ──────────────────────────────────────────────
# parse_gbnf: non-terminal references
# ──────────────────────────────────────────────

def test_nonterminal_reference():
    terminals, prods = parse_gbnf('root ::= item\nitem ::= "x"')
    assert prods["root"] == {("item",)}
    assert len(terminals) == 1
    t = list(terminals.keys())[0]
    assert prods["item"] == {(t,)}


def test_nonterminal_sequence():
    terminals, prods = parse_gbnf('root ::= item item\nitem ::= "x"')
    assert prods["root"] == {("item", "item")}


# ──────────────────────────────────────────────
# parse_gbnf: quantifiers on non-terminals
# ──────────────────────────────────────────────

def test_nonterminal_star():
    terminals, prods = parse_gbnf('root ::= item*\nitem ::= "x"')
    assert "root" in prods
    root_alts = prods["root"]
    assert len(root_alts) == 1, f"root alts: {root_alts}"
    (root_tup,) = root_alts
    assert len(root_tup) == 1, f"root tuple: {root_tup}"
    aux = root_tup[0]
    assert aux != "item", "root should reference an auxiliary rule, not item directly"
    alts = prods.get(aux, set())
    assert () in alts, f"aux {aux} missing ε alternative"
    assert any("item" in tup for tup in alts), f"aux {aux} missing recursive alternative with item"


def test_nonterminal_plus():
    terminals, prods = parse_gbnf('root ::= item+\nitem ::= "x"')
    assert "root" in prods
    root_alts = prods["root"]
    assert len(root_alts) == 1
    (root_tup,) = root_alts
    assert len(root_tup) == 1
    aux = root_tup[0]
    alts = prods.get(aux, set())
    assert any(tup == ("item",) for tup in alts), f"aux {aux} missing base (item) alternative"
    assert any("item" in tup and aux in tup for tup in alts), f"aux {aux} missing recursive alternative"
    assert () not in alts, f"aux {aux} must not have ε (item+ means one or more)"


def test_nonterminal_optional():
    terminals, prods = parse_gbnf('root ::= item?\nitem ::= "x"')
    assert "root" in prods
    root_alts = prods["root"]
    if root_alts == {(), ("item",)}:
        pass  # direct encoding
    else:
        assert len(root_alts) == 1
        (root_tup,) = root_alts
        assert len(root_tup) == 1
        aux = root_tup[0]
        alts = prods.get(aux, set())
        assert () in alts, f"aux {aux} missing ε"
        assert any("item" in tup for tup in alts), f"aux {aux} missing item alternative"


# ──────────────────────────────────────────────
# parse_gbnf: comments
# ──────────────────────────────────────────────

def test_comment_inline():
    terminals1, prods1 = parse_gbnf('root ::= "a"')
    terminals2, prods2 = parse_gbnf('root ::= "a" # this is a comment')
    assert set(terminals1.values()) == set(terminals2.values())
    assert prods1["root"] == prods2["root"]


def test_comment_full_line():
    terminals, prods = parse_gbnf('# full line comment\nroot ::= "a"')
    assert "root" in prods, f"root missing, prods={prods}"


# ──────────────────────────────────────────────
# parse_gbnf: grouping that creates auxiliary rules
# ──────────────────────────────────────────────

def test_group_with_nonterminal_alternative():
    terminals, prods = parse_gbnf('root ::= (a | b) c\na ::= "x"\nb ::= "y"\nc ::= "z"')
    for tup in prods["root"]:
        assert "c" in tup or any("c" in prods.get(sym, set()) for sym in tup), \
            f"c missing from root prod {tup}"


# ──────────────────────────────────────────────
# parse_gbnf: real grammars from gbnf.md
# ──────────────────────────────────────────────

CHESS_GBNF = """
root ::= (
    "1. " move " " move "\\n"
    ([1-9] [0-9]? ". " move " " move "\\n")+
)

move ::= (pawn | nonpawn | castle) [+#]?

pawn ::= [a-h] ([1-8] | [a-h] "x" [a-h] [1-8] | [a-h] "x" [a-h] "8" "=" piece)

nonpawn ::= piece ([a-h]? [1-8]? "x")? [a-h] [1-8] ("+" | "#")?

castle ::= ("O-O" | "O-O-O") [+#]?

piece ::= [NBKQR]
"""

LIST_GBNF = """
root ::= ("- " item)+
item ::= [^\\n]+ "\\n"
"""


def test_list_grammar():
    terminals, prods = parse_gbnf(LIST_GBNF)
    assert "root" in prods
    assert "item" in prods


def test_closure_check():
    """Every symbol referenced in a production must be a known non-terminal or terminal."""
    terminals, prods = parse_gbnf(LIST_GBNF)
    known_nt = set(prods.keys())
    known_t = set(terminals.keys()) | {""}
    for lhs, alts in prods.items():
        for tup in alts:
            for sym in tup:
                assert sym in known_nt or sym in known_t or sym == '""', \
                    f"unknown symbol {sym!r} in {lhs} -> {tup}"


def test_chess_grammar():
    terminals, prods = parse_gbnf(CHESS_GBNF)
    known_nt = set(prods.keys())
    known_t = set(terminals.keys())

    def find_t(should_match, should_not=()):
        for lbl, rx in terminals.items():
            try:
                pat = re.compile(rx)
            except re.error:
                continue
            if all(pat.fullmatch(s) for s in should_match) and \
               not any(pat.fullmatch(s) for s in should_not):
                return lbl
        return None

    # piece
    t_piece = find_t(["N", "B", "K", "Q", "R"], ["a", "P", "1", ""])
    assert t_piece, "no terminal matching [NBKQR]"
    assert prods["piece"] == {(t_piece,)}, f"piece: {prods['piece']}"

    # castle
    t_castle = find_t(
        ["O-O", "O-O-O", "O-O+", "O-O#", "O-O-O+"],
        ["O", "O-", "O-O-O-O", "O-O++"],
    )
    assert t_castle, "no terminal matching castle notation"
    assert prods["castle"] == {(t_castle,)}, f"castle: {prods['castle']}"

    # move: one production (move_group_aux, check_terminal)
    assert len(prods["move"]) == 1, f"move has {len(prods['move'])} alts"
    (move_tup,) = prods["move"]
    assert len(move_tup) == 2, f"move production length: {move_tup}"
    move_aux, move_check_t = move_tup
    assert move_aux in known_nt, f"move first sym not an NT: {move_aux!r}"
    assert move_check_t in known_t, f"move second sym not a terminal: {move_check_t!r}"
    t_check_rx = terminals[move_check_t]
    assert _matches(t_check_rx, "+") and _matches(t_check_rx, "#") and _matches(t_check_rx, ""), \
        f"move check terminal should match +, #, '': {t_check_rx!r}"
    assert prods[move_aux] == {("pawn",), ("nonpawn",), ("castle",)}, \
        f"move aux {move_aux} prods: {prods[move_aux]}"

    # pawn: one production (file_terminal, alternatives_aux)
    assert len(prods["pawn"]) == 1, f"pawn has {len(prods['pawn'])} alts"
    (pawn_tup,) = prods["pawn"]
    assert len(pawn_tup) == 2, f"pawn production length: {pawn_tup}"
    pawn_file_t, pawn_aux = pawn_tup
    assert pawn_file_t in known_t, f"pawn first sym not a terminal: {pawn_file_t!r}"
    t_file_rx = terminals[pawn_file_t]
    assert _matches(t_file_rx, "a") and _matches(t_file_rx, "h") and not _matches(t_file_rx, "i"), \
        f"pawn file terminal bad: {t_file_rx!r}"
    assert pawn_aux in known_nt, f"pawn second sym not an NT: {pawn_aux!r}"
    pawn_alts = prods[pawn_aux]
    assert len(pawn_alts) == 3, f"pawn aux has {len(pawn_alts)} alts (expected 3)"
    assert any("piece" in tup for tup in pawn_alts), \
        f"no pawn alt references piece: {pawn_alts}"

    # nonpawn: one production (piece, move_terminal)
    assert len(prods["nonpawn"]) == 1, f"nonpawn has {len(prods['nonpawn'])} alts"
    (nonpawn_tup,) = prods["nonpawn"]
    assert len(nonpawn_tup) == 2, f"nonpawn production length: {nonpawn_tup}"
    assert nonpawn_tup[0] == "piece", f"nonpawn first sym: {nonpawn_tup[0]!r}"
    nonpawn_move_t = nonpawn_tup[1]
    assert nonpawn_move_t in known_t, "nonpawn second sym not a terminal"
    t_npm_rx = terminals[nonpawn_move_t]
    assert _matches(t_npm_rx, "e4") and _matches(t_npm_rx, "axe4") and \
           _matches(t_npm_rx, "e4+") and _matches(t_npm_rx, "a1xe4#") and \
           not _matches(t_npm_rx, "e9") and not _matches(t_npm_rx, ""), \
        f"nonpawn move terminal bad: {t_npm_rx!r}"

    # root: one production → aux → contains move twice
    assert len(prods["root"]) == 1, f"root has {len(prods['root'])} alts"
    (root_tup,) = prods["root"]
    assert len(root_tup) == 1 and root_tup[0] in known_nt, \
        f"root should reference one aux NT: {root_tup}"
    root_aux_alts = prods[root_tup[0]]
    assert any(tup.count("move") >= 2 for tup in root_aux_alts), \
        f"root aux has no alt with 2+ move refs: {root_aux_alts}"

    # closure
    for lhs, alts in prods.items():
        for tup in alts:
            for sym in tup:
                assert sym in known_nt or sym in known_t, \
                    f"unknown symbol {sym!r} in {lhs} -> {tup}"
