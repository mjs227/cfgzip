
import re
import sys
from cfgzip.preprocessing.parse_gbnf import parse_gbnf, gbnf_to_regex

failures = []


def check(name, actual, expected, *, exact=True):
    if exact:
        ok = actual == expected
    else:
        ok = expected == actual  # same but kept for extensibility
    if not ok:
        failures.append(name)
        print(f"FAIL: {name}")
        print(f"  expected: {expected!r}")
        print(f"  actual:   {actual!r}")
    else:
        print(f"PASS: {name}")
    return ok


def check_regex(name, regex, should_match, should_not_match):
    try:
        pat = re.compile(regex)
    except re.error as e:
        failures.append(name)
        print(f"FAIL: {name} — invalid regex {regex!r}: {e}")
        return
    bad = []
    for s in should_match:
        if not pat.fullmatch(s):
            bad.append(f"no match: {s!r}")
    for s in should_not_match:
        if pat.fullmatch(s):
            bad.append(f"wrong match: {s!r}")
    if bad:
        failures.append(name)
        print(f"FAIL: {name} (regex={regex!r})")
        for b in bad:
            print(f"  {b}")
    else:
        print(f"PASS: {name}")


# ──────────────────────────────────────────────
# gbnf_to_regex unit tests
# ──────────────────────────────────────────────

def test_gbnf_to_regex_string_simple():
    r = gbnf_to_regex('"abc"')
    check("gbnf_to_regex: simple string", r, "abc")


def test_gbnf_to_regex_string_special_chars():
    r = gbnf_to_regex('"a.b"')
    check_regex("gbnf_to_regex: escaped dot", r, ["a.b"], ["axb"])


def test_gbnf_to_regex_char_class():
    r = gbnf_to_regex('[a-z]')
    check_regex("gbnf_to_regex: char class [a-z]", r, ["a", "z"], ["A", "0"])


def test_gbnf_to_regex_char_class_negated():
    r = gbnf_to_regex(r'[^\n]')
    check_regex("gbnf_to_regex: negated char class [^\\n]", r, ["a", " "], ["\n"])


def test_gbnf_to_regex_string_plus():
    # "abc"+ should become (?:abc)+ — matches one or more "abc"
    r = gbnf_to_regex('"abc"+')
    check_regex("gbnf_to_regex: string+", r, ["abc", "abcabc"], ["ab", ""])


def test_gbnf_to_regex_charclass_plus():
    r = gbnf_to_regex('[a-z]+')
    check_regex("gbnf_to_regex: [a-z]+", r, ["a", "abc"], ["", "A"])


def test_gbnf_to_regex_charclass_star():
    r = gbnf_to_regex('[a-z]*')
    check_regex("gbnf_to_regex: [a-z]*", r, ["", "a", "abc"], ["A"])


def test_gbnf_to_regex_charclass_optional():
    r = gbnf_to_regex('[0-9]?')
    check_regex("gbnf_to_regex: [0-9]?", r, ["", "5"], ["55"])


def test_gbnf_to_regex_concat_string_charclass():
    r = gbnf_to_regex('"ab" [0-9]')
    check_regex("gbnf_to_regex: concat str+class", r, ["ab5"], ["ab", "5"])


def test_gbnf_to_regex_alternation():
    r = gbnf_to_regex('("a" | "b")')
    check_regex("gbnf_to_regex: alternation (a|b)", r, ["a", "b"], ["c", "ab"])


def test_gbnf_to_regex_group_plus():
    r = gbnf_to_regex('("a" | "b")+')
    check_regex("gbnf_to_regex: (a|b)+", r, ["a", "b", "ab", "ba", "aab"], ["", "c"])


def test_gbnf_to_regex_nested():
    r = gbnf_to_regex('([a-z] [0-9]*)')
    check_regex("gbnf_to_regex: nested [a-z][0-9]*", r, ["a", "a1", "a123"], ["1", ""])


def test_gbnf_to_regex_escape_hex():
    r = gbnf_to_regex(r'"\x41"')   # \x41 = 'A'
    check_regex("gbnf_to_regex: hex escape \\x41", r, ["A"], ["a"])


# ──────────────────────────────────────────────
# parse_gbnf: terminal extraction
# ──────────────────────────────────────────────

def test_terminal_extraction_simple():
    terminals, prods = parse_gbnf('root ::= "hello"')
    # should have one terminal T0 with regex matching "hello"
    assert len(terminals) == 1, f"expected 1 terminal, got {len(terminals)}"
    t_label = list(terminals.keys())[0]
    check_regex("terminal extraction: 'hello'", terminals[t_label], ["hello"], ["hell", "helloo"])
    # root production should reference that terminal
    check("terminal extraction: root prod", prods["root"], {(t_label,)})


def test_terminal_extraction_reuse():
    terminals, prods = parse_gbnf('root ::= "a" "a"')
    # "a" used twice but should only be one terminal
    check("terminal extraction: reuse", len(terminals), 1, exact=False)
    t = list(terminals.keys())[0]
    check("terminal extraction: reuse prod", prods["root"], {(t,)})


def test_terminal_extraction_char_class():
    terminals, prods = parse_gbnf('root ::= [a-z]+')
    assert len(terminals) == 1
    t = list(terminals.keys())[0]
    check_regex("terminal extraction: [a-z]+", terminals[t], ["a", "abc"], ["", "A"])
    check("terminal extraction: [a-z]+ prod", prods["root"], {(t,)})


def test_terminal_mixed():
    terminals, prods = parse_gbnf('root ::= "id_" [a-z]+')
    # both parts should merge into one terminal regex
    assert len(terminals) == 1, f"expected 1 terminal, got {len(terminals)}: {terminals}"
    t = list(terminals.keys())[0]
    check_regex("terminal extraction: mixed", terminals[t], ["id_a", "id_abc"], ["id_", "_a"])
    check("terminal extraction: mixed prod", prods["root"], {(t,)})


# ──────────────────────────────────────────────
# parse_gbnf: alternation (the | operator)
# ──────────────────────────────────────────────

def test_alternation_basic():
    terminals, prods = parse_gbnf('root ::= "a" | "b"')
    assert len(terminals) == 2
    t_a = next(k for k, v in terminals.items() if re.fullmatch(v, "a"))
    t_b = next(k for k, v in terminals.items() if re.fullmatch(v, "b"))
    check("alternation basic", prods["root"], {(t_a,), (t_b,)})


def test_alternation_multi_symbol():
    terminals, prods = parse_gbnf('root ::= "a" "b" | "c"')
    t_ab_regex_ok = any(
        re.fullmatch(v, "a") for v in terminals.values()
    )
    # productions should have 2 alternatives: one with two terminal refs, one with one
    sizes = sorted(len(tup) for tup in prods["root"])
    check("alternation multi-symbol sizes", sizes, [1, 1])


# ──────────────────────────────────────────────
# parse_gbnf: non-terminal references
# ──────────────────────────────────────────────

def test_nonterminal_reference():
    terminals, prods = parse_gbnf('root ::= item\nitem ::= "x"')
    # root -> item, item -> T0
    check("nonterminal ref: root prod", prods["root"], {("item",)})
    assert len(terminals) == 1
    t = list(terminals.keys())[0]
    check("nonterminal ref: item prod", prods["item"], {(t,)})


def test_nonterminal_sequence():
    terminals, prods = parse_gbnf('root ::= item item\nitem ::= "x"')
    check("nonterminal sequence: root", prods["root"], {("item", "item")})


# ──────────────────────────────────────────────
# parse_gbnf: quantifiers on non-terminals
# ──────────────────────────────────────────────

def _has_epsilon_and_recurse(prods, aux_sym, base_sym):
    """Check that aux_sym has an ε alternative and at least one recursive alternative."""
    alts = prods.get(aux_sym, set())
    has_empty = () in alts
    has_recurse = any(base_sym in tup for tup in alts)
    return has_empty, has_recurse


def test_nonterminal_star():
    # root ::= item* => new rule __rep_item -> ε | item __rep_item (or similar)
    terminals, prods = parse_gbnf('root ::= item*\nitem ::= "x"')
    # root should reference a single auxiliary non-terminal (not 'item' directly with *)
    assert "root" in prods
    root_alts = prods["root"]
    # root should have exactly one alternative which is a single auxiliary symbol
    assert len(root_alts) == 1, f"root alts: {root_alts}"
    (root_tup,) = root_alts
    assert len(root_tup) == 1, f"root tuple: {root_tup}"
    aux = root_tup[0]
    assert aux != "item", "root should reference an auxiliary rule, not item directly"
    has_empty, has_recurse = _has_epsilon_and_recurse(prods, aux, "item")
    check(f"item* aux {aux}: has ε", has_empty, True)
    check(f"item* aux {aux}: has recurse with item", has_recurse, True)


def test_nonterminal_plus():
    terminals, prods = parse_gbnf('root ::= item+\nitem ::= "x"')
    assert "root" in prods
    root_alts = prods["root"]
    assert len(root_alts) == 1
    (root_tup,) = root_alts
    assert len(root_tup) == 1
    aux = root_tup[0]
    alts = prods.get(aux, set())
    # Must have: item | item aux  (one or more)
    has_single = any(tup == ("item",) for tup in alts)
    has_double = any("item" in tup and aux in tup for tup in alts)
    check(f"item+ aux {aux}: has base (item)", has_single, True)
    check(f"item+ aux {aux}: has recursive", has_double, True)
    # Must NOT have ε
    check(f"item+ aux {aux}: no ε", () in alts, False)


def test_nonterminal_optional():
    terminals, prods = parse_gbnf('root ::= item?\nitem ::= "x"')
    assert "root" in prods
    root_alts = prods["root"]
    # Either root itself has ε and item, or delegates to an aux
    if root_alts == {(), ("item",)}:
        check("item?: root has ε and item directly", True, True)
    else:
        assert len(root_alts) == 1
        (root_tup,) = root_alts
        assert len(root_tup) == 1
        aux = root_tup[0]
        alts = prods.get(aux, set())
        check(f"item? aux {aux}: has ε", () in alts, True)
        check(f"item? aux {aux}: has item", any("item" in tup for tup in alts), True)


# ──────────────────────────────────────────────
# parse_gbnf: comments
# ──────────────────────────────────────────────

def test_comment_inline():
    terminals1, prods1 = parse_gbnf('root ::= "a"')
    terminals2, prods2 = parse_gbnf('root ::= "a" # this is a comment')
    check("inline comment: same terminals", set(terminals1.values()) == set(terminals2.values()), True)
    check("inline comment: same prods", prods1["root"] == prods2["root"], True)


def test_comment_full_line():
    terminals, prods = parse_gbnf('# full line comment\nroot ::= "a"')
    assert "root" in prods, f"root missing, prods={prods}"
    check("full-line comment: root present", "root" in prods, True)


# ──────────────────────────────────────────────
# parse_gbnf: grouping that creates auxiliary rules
# ──────────────────────────────────────────────

def test_group_with_nonterminal_alternative():
    # (a | b) c  => root has two alternatives: a c and b c
    # or root -> aux c, aux -> a | b
    terminals, prods = parse_gbnf('root ::= (a | b) c\na ::= "x"\nb ::= "y"\nc ::= "z"')
    # Every production of root should eventually "include" c
    for tup in prods["root"]:
        assert "c" in tup or any("c" in prods.get(sym, set()) for sym in tup), \
            f"c missing from root prod {tup}"
    print(f"PASS: group with nonterminal alternative (root={prods['root']})")


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
    try:
        terminals, prods = parse_gbnf(LIST_GBNF)
        assert "root" in prods
        assert "item" in prods
        print(f"PASS: list grammar (terminals={list(terminals.keys())}, rules={list(prods.keys())})")
    except Exception as e:
        failures.append("list grammar")
        print(f"FAIL: list grammar — {e}")


def test_chess_grammar():
    try:
        terminals, prods = parse_gbnf(CHESS_GBNF)
        known_nt = set(prods.keys())
        known_t  = set(terminals.keys())

        def find_t(should_match, should_not=()):
            for lbl, rx in terminals.items():
                try: pat = re.compile(rx)
                except re.error: continue
                if all(pat.fullmatch(s) for s in should_match) and \
                   not any(pat.fullmatch(s) for s in should_not):
                    return lbl
            return None

        # ── piece ────────────────────────────────────────────────
        t_piece = find_t(["N", "B", "K", "Q", "R"], ["a", "P", "1", ""])
        assert t_piece, "no terminal matching [NBKQR]"
        assert prods["piece"] == {(t_piece,)}, f"piece: {prods['piece']}"

        # ── castle ───────────────────────────────────────────────
        t_castle = find_t(
            ["O-O", "O-O-O", "O-O+", "O-O#", "O-O-O+"],
            ["O", "O-", "O-O-O-O", "O-O++"],
        )
        assert t_castle, "no terminal matching castle notation"
        assert prods["castle"] == {(t_castle,)}, f"castle: {prods['castle']}"

        # ── move: one production (move_group_aux, check_terminal) ─
        assert len(prods["move"]) == 1, f"move has {len(prods['move'])} alts"
        (move_tup,) = prods["move"]
        assert len(move_tup) == 2, f"move production length: {move_tup}"
        move_aux, move_check_t = move_tup
        assert move_aux in known_nt, f"move first sym not an NT: {move_aux!r}"
        assert move_check_t in known_t, f"move second sym not a terminal: {move_check_t!r}"
        t_check_rx = terminals[move_check_t]
        assert re.fullmatch(t_check_rx, "+") and re.fullmatch(t_check_rx, "#") \
               and re.fullmatch(t_check_rx, ""), \
               f"move check terminal should match +, #, '': {t_check_rx!r}"
        assert prods[move_aux] == {("pawn",), ("nonpawn",), ("castle",)}, \
            f"move aux {move_aux} prods: {prods[move_aux]}"

        # ── pawn: one production (file_terminal, alternatives_aux) ─
        assert len(prods["pawn"]) == 1, f"pawn has {len(prods['pawn'])} alts"
        (pawn_tup,) = prods["pawn"]
        assert len(pawn_tup) == 2, f"pawn production length: {pawn_tup}"
        pawn_file_t, pawn_aux = pawn_tup
        assert pawn_file_t in known_t, f"pawn first sym not a terminal: {pawn_file_t!r}"
        t_file_rx = terminals[pawn_file_t]
        assert re.fullmatch(t_file_rx, "a") and re.fullmatch(t_file_rx, "h") \
               and not re.fullmatch(t_file_rx, "i"), \
               f"pawn file terminal bad: {t_file_rx!r}"
        assert pawn_aux in known_nt, f"pawn second sym not an NT: {pawn_aux!r}"
        pawn_alts = prods[pawn_aux]
        assert len(pawn_alts) == 3, f"pawn aux has {len(pawn_alts)} alts (expected 3)"
        assert any("piece" in tup for tup in pawn_alts), \
            f"no pawn alt references piece: {pawn_alts}"

        # ── nonpawn: one production (piece, move_terminal) ───────
        assert len(prods["nonpawn"]) == 1, f"nonpawn has {len(prods['nonpawn'])} alts"
        (nonpawn_tup,) = prods["nonpawn"]
        assert len(nonpawn_tup) == 2, f"nonpawn production length: {nonpawn_tup}"
        assert nonpawn_tup[0] == "piece", f"nonpawn first sym: {nonpawn_tup[0]!r}"
        nonpawn_move_t = nonpawn_tup[1]
        assert nonpawn_move_t in known_t, "nonpawn second sym not a terminal"
        t_npm_rx = terminals[nonpawn_move_t]
        assert re.fullmatch(t_npm_rx, "e4") and re.fullmatch(t_npm_rx, "axe4") \
               and re.fullmatch(t_npm_rx, "e4+") and re.fullmatch(t_npm_rx, "a1xe4#") \
               and not re.fullmatch(t_npm_rx, "e9") and not re.fullmatch(t_npm_rx, ""), \
               f"nonpawn move terminal bad: {t_npm_rx!r}"

        # ── root: one production → aux → contains move twice ─────
        assert len(prods["root"]) == 1, f"root has {len(prods['root'])} alts"
        (root_tup,) = prods["root"]
        assert len(root_tup) == 1 and root_tup[0] in known_nt, \
            f"root should reference one aux NT: {root_tup}"
        root_aux_alts = prods[root_tup[0]]
        assert any(tup.count("move") >= 2 for tup in root_aux_alts), \
            f"root aux has no alt with 2+ move refs: {root_aux_alts}"

        # ── closure ──────────────────────────────────────────────
        for lhs, alts in prods.items():
            for tup in alts:
                for sym in tup:
                    assert sym in known_nt or sym in known_t, \
                        f"unknown symbol {sym!r} in {lhs} -> {tup}"

        print("PASS: chess grammar (exact production and terminal check)")
    except AssertionError as e:
        failures.append("chess grammar")
        print(f"FAIL: chess grammar — AssertionError: {e}")
    except Exception as e:
        failures.append("chess grammar")
        print(f"FAIL: chess grammar — {type(e).__name__}: {e}")


def test_closure_check():
    """Every symbol referenced in a production must be a known non-terminal or terminal."""
    try:
        terminals, prods = parse_gbnf(LIST_GBNF)
        known_nt = set(prods.keys())
        known_t = set(terminals.keys()) | {""}  # ε / empty
        for lhs, alts in prods.items():
            for tup in alts:
                for sym in tup:
                    assert sym in known_nt or sym in known_t or sym == '""', \
                        f"unknown symbol {sym!r} in {lhs} -> {tup}"
        print("PASS: closure check on list grammar")
    except Exception as e:
        failures.append("closure check")
        print(f"FAIL: closure check — {e}")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"\nRunning {len(tests)} tests...\n")
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL: {t.__name__} — AssertionError: {e}")
        except Exception as e:
            failures.append(t.__name__)
            print(f"FAIL: {t.__name__} — Exception: {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("All tests passed.")
