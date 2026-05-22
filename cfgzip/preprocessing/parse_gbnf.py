
import re
from typing import Dict, List, Tuple, Set
from cfgzip.preprocessing.utils import sdict_add


# this file is (mostly) written by Claude

# ── regex conversion ────────────────────────────────────────────────────────


# converts GBNF string literal content to a Python regex fragment---passes through recognized
# escape sequences (\n, \t, \xNN, \uNNNN, \UNNNNNNNN) unchanged since Python regex also understands
# them; re.escapes plain chars that would otherwise be interpreted as regex operators
def _escape_literal(s: str) -> str:
    result, i = [], 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in ('n', 't', 'r', 'f', 'v', '\\', '"', "'", '/'):
                result.append(s[i:i + 2])
                i += 2
            elif nxt == 'x' and i + 3 < len(s):
                result.append(s[i:i + 4])
                i += 4
            elif nxt == 'u' and i + 5 < len(s):
                result.append(s[i:i + 6])
                i += 6
            elif nxt == 'U' and i + 9 < len(s):
                result.append(s[i:i + 10])
                i += 10
            else:
                result.append(re.escape(s[i]))
                i += 1
        else:
            result.append(re.escape(s[i]))
            i += 1
    return ''.join(result)


# wraps a regex fragment so a following quantifier applies to the whole atom, not just its last char
# e.g. "abc"+ must become (?:abc)+ not abc+; [a-z]+ and single chars are already atomic so left as-is;
# (…) produced by _gbnf_to_regex_rec are capturing groups GBNF doesn't use, so convert to non-capturing
def _wrap_for_quant(s: str) -> str:
    if len(s) == 1:
        return s
    if s[0] == '[' and s[-1] == ']':
        return s
    if s[0] == '(' and s[-1] == ')':
        return f'(?:{s[1:-1]})'
    return f'(?:{s})'


# breaks the contents of a GBNF (…) expression into a flat list of constituent token strings:
# "…" string literals, […] char classes, (…) sub-groups, operator chars (|, *, +, ?),
# and (if match_nonterminals=True) nonterminal names as full strings instead of char-by-char
def _extract_groups(ebnf_string: str, match_nonterminals: bool = False) -> List[str]:
    match_bracket, match_string, match_nt, paren_cnt, esc_mode = False, False, False, 0, False
    ebnf_string, groups = ebnf_string[1:-1], []  # strip outer parens before scanning

    for i, char in enumerate(ebnf_string):
        if match_nt and not (char.isalnum() or char == '_'):  # end of nonterminal name
            match_nt = False
            groups[-1] = ebnf_string[groups[-1]:i]  # replace start index with the full name slice
        if char == '\\':
            esc_mode = not esc_mode
        elif esc_mode:
            esc_mode = False
        elif match_string:
            if char == '"':
                groups[-1] = ebnf_string[groups[-1]:(i + 1)]  # replace start index with full "…" slice
                match_string = False
        elif match_bracket:
            if char == ']':
                groups[-1] = ebnf_string[groups[-1]:(i + 1)]  # replace start index with full […] slice
                match_bracket = False
        elif paren_cnt > 0:
            if char == ')':
                paren_cnt -= 1
                if paren_cnt == 0: groups[-1] = ebnf_string[groups[-1]:(i + 1)]  # full (…) slice
            elif char == '(':
                paren_cnt += 1
        elif char == '"':
            match_string = True
            groups.append(i)  # store start index; replaced with slice when closing " is found
        elif char == '[':
            match_bracket = True
            groups.append(i)
        elif char == '(':
            paren_cnt = 1
            groups.append(i)
        elif char.isalnum() or char == '_':
            if not match_nt:
                if match_nonterminals:
                    match_nt = True
                    groups.append(i)  # store start index; replaced with full name when name ends
                else:
                    groups.append(char)  # single-char mode: each alnum char is its own token
        elif not char.isspace():
            groups.append(char)  # operators: |, *, +, ?

    if match_nt: groups[-1] = ebnf_string[groups[-1]:]  # nonterminal name that runs to end of string

    return groups


# recursively converts a GBNF atom to a Python regex string
# base cases: char class [..] (returned as-is), string literal "…" (content escaped via _escape_literal),
# group (…) (inner tokens converted recursively; quantifiers wrapped via _wrap_for_quant)
def _gbnf_to_regex_rec(ebnf_string: str) -> str:
    ebnf_string = ebnf_string.strip()

    if len(ebnf_string) < 2 or (ebnf_string[0] == '[' and ebnf_string[-1] == ']'):
        return ebnf_string.replace('\\"', '"').replace("\\'", "'")
    if ebnf_string[0] == ebnf_string[-1] == '"':
        return _escape_literal(ebnf_string[1:-1])
    if ebnf_string[0] == '(' and ebnf_string[-1] == ')':
        groups = _extract_groups(ebnf_string)
        parts, i = [], 0
        while i < len(groups):
            converted = _gbnf_to_regex_rec(groups[i])
            if i + 1 < len(groups) and groups[i + 1] in ('*', '+', '?'):
                parts.append(_wrap_for_quant(converted) + groups[i + 1])
                i += 2
            else:
                parts.append(converted)
                i += 1
        return f'({"".join(parts)})'

    raise Exception(f'unrecognized gbnf atom: {ebnf_string!r}')


# converts a GBNF pattern string (the RHS of a pure-terminal rule) to a Python regex---
# wraps in (…) so _gbnf_to_regex_rec always sees a group at the top level, then strips it
def gbnf_to_regex(ebnf_string: str) -> str:
    return _gbnf_to_regex_rec(rf'({ebnf_string})')[1:-1]


# ── RHS tokenizer ────────────────────────────────────────────────────────────


# reads one optional quantifier char (*, +, ?) at position j; returns (quant, new_j) or ('', j)
def _read_quant(s: str, j: int) -> Tuple[str, int]:
    if j < len(s) and s[j] in ('*', '+', '?'):
        return s[j], j + 1
    return '', j


# tokenizes a GBNF RHS into typed tokens: ('str', text, quant), ('cls', text, quant),
# ('group', text, quant), ('nt', name, quant), or ('alt',) for | separators---
# groups are captured with balanced paren tracking that respects nested strings and char classes
def _tokenize(rhs: str) -> List[Tuple]:
    tokens, i, n = [], 0, len(rhs)
    while i < n:
        c = rhs[i]
        if c.isspace():
            i += 1
        elif c == '#':
            break  # inline comment: discard rest of line
        elif c == '|':
            tokens.append(('alt',))
            i += 1
        elif c == '"':
            j = i + 1
            while j < n:
                if rhs[j] == '\\': j += 2
                elif rhs[j] == '"': j += 1; break
                else: j += 1
            text = rhs[i:j]
            q, j = _read_quant(rhs, j)
            tokens.append(('str', text, q))
            i = j
        elif c == '[':
            j = i + 1
            while j < n:
                if rhs[j] == '\\': j += 2
                elif rhs[j] == ']': j += 1; break
                else: j += 1
            text = rhs[i:j]
            q, j = _read_quant(rhs, j)
            tokens.append(('cls', text, q))
            i = j
        elif c == '(':
            depth, j, esc, in_str, in_cls = 1, i + 1, False, False, False
            while j < n and depth > 0:
                ch = rhs[j]
                if esc: esc = False
                elif ch == '\\': esc = True
                elif in_str:
                    if ch == '"': in_str = False
                elif in_cls:
                    if ch == ']': in_cls = False
                elif ch == '"': in_str = True
                elif ch == '[': in_cls = True
                elif ch == '(': depth += 1
                elif ch == ')': depth -= 1
                j += 1
            text = rhs[i:j]
            q, j = _read_quant(rhs, j)
            tokens.append(('group', text, q))
            i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < n and (rhs[j].isalnum() or rhs[j] in '_-'):
                j += 1
            text = rhs[i:j]
            q, j = _read_quant(rhs, j)
            tokens.append(('nt', text, q))
            i = j
        else:
            i += 1
    return tokens


# returns True if a token contains no nonterminal references and can be expressed as a pure regex---
# 'str' and 'cls' tokens are always terminal; a 'group' token is terminal only if all its inner tokens
# (recursively) are also terminal---used to decide whether to merge tokens into a single terminal label
def _is_terminal_tok(tok: Tuple) -> bool:
    kind = tok[0]
    if kind in ('str', 'cls'):
        return True
    if kind == 'group':
        inner = tok[1][1:-1]
        return all(_is_terminal_tok(t) for t in _tokenize(inner) if t[0] != 'alt')
    return False


# ── parse_gbnf internals ─────────────────────────────────────────────────────


# registers a GBNF pattern string as a labeled terminal (__T0, __T1, …) and returns the label---
# deduplicates: the same pattern always gets the same label; the '""' sentinel is always present
# in terminals during construction, so len(terminals)-1 gives the count of real terminals
def _add_terminal(terminals: Dict, pattern: str) -> str:
    if pattern not in terminals:
        terminals[pattern] = f'__T{len(terminals) - 1}'
    return terminals[pattern]


# creates an auxiliary nonterminal __repN to represent repetition of `base` by `quant`---
# unrolled CFG productions:  A* → __repN ::= ε | A __repN
#                             A+ → __repN ::= A | A __repN
#                             A? → __repN ::= ε | A
def _make_quant_aux(
        base: str, quant: str,
        productions: Dict, aux_counter: List
) -> str:
    aux = f'__rep{aux_counter[0]}'
    aux_counter[0] += 1
    if quant == '*':
        sdict_add(productions, aux, ())
        sdict_add(productions, aux, (base, aux))
    elif quant == '+':
        sdict_add(productions, aux, (base,))
        sdict_add(productions, aux, (base, aux))
    elif quant == '?':
        sdict_add(productions, aux, ())
        sdict_add(productions, aux, (base,))
    return aux


# processes one alternative (tokens between | separators) from a GBNF rule RHS---
# consecutive terminal tokens are accumulated in a run and flushed as a single terminal label
# when a nonterminal or nonterminal-containing group is encountered, or at the end of the alt;
# nonterminals with quantifiers become new __repN aux rules; groups with nonterminal content
# become new __grpN aux rules parsed recursively via _parse_rhs
def _process_alt(
        tokens: List[Tuple],
        terminals: Dict, productions: Dict, aux_counter: List
) -> List[str]:
    syms: List[str] = []
    terminal_run: List[Tuple] = []

    def flush() -> None:
        if not terminal_run:
            return
        pattern = ' '.join(t[1] + t[2] for t in terminal_run)  # gbnf fragment: text + quant
        syms.append(_add_terminal(terminals, pattern))
        terminal_run.clear()

    for tok in tokens:
        kind = tok[0]
        if _is_terminal_tok(tok):
            terminal_run.append(tok)
        elif kind == 'nt':
            _, name, quant = tok
            flush()
            syms.append(_make_quant_aux(name, quant, productions, aux_counter) if quant else name)
        elif kind == 'group':
            _, text, quant = tok
            flush()
            inner = text[1:-1]
            aux = f'__grp{aux_counter[0]}'
            aux_counter[0] += 1
            _parse_rhs(aux, inner, terminals, productions, aux_counter)
            if quant:
                aux = _make_quant_aux(aux, quant, productions, aux_counter)
            syms.append(aux)

    flush()
    return syms


# splits a full GBNF RHS by | into alternatives and adds one production per alternative to lhs---
# an empty alternative (e.g. the leading | in  rule ::= | "a" | "b") produces the ε production ()
def _parse_rhs(
        lhs: str, rhs: str,
        terminals: Dict, productions: Dict, aux_counter: List
) -> None:
    tokens = _tokenize(rhs)
    alts: List[List[Tuple]] = [[]]
    for tok in tokens:
        if tok[0] == 'alt':
            alts.append([])
        else:
            alts[-1].append(tok)

    productions.setdefault(lhs, set())
    for alt in alts:
        if not alt:
            sdict_add(productions, lhs, ())
        else:
            sdict_add(productions, lhs, tuple(_process_alt(alt, terminals, productions, aux_counter)))


# GBNF rules can span multiple lines when the RHS contains unclosed parentheses---
# joins such continuation lines with a space before passing them to the parser;
# tracks paren depth respecting nested strings and char classes; normal single-line rules
# return to depth 0 at line end and pass through unchanged
def _join_lines(lines: List[str]) -> List[str]:
    result, parts = [], []
    depth, in_str, in_cls, esc = 0, False, False, False

    for line in lines:
        parts.append(line)
        for c in line:
            if esc: esc = False
            elif c == '\\': esc = True
            elif in_str:
                if c == '"': in_str = False
            elif in_cls:
                if c == ']': in_cls = False
            elif c == '"': in_str = True
            elif c == '[': in_cls = True
            elif c == '(': depth += 1
            elif c == ')': depth -= 1
        if depth <= 0:
            result.append(' '.join(parts))
            parts = []
            depth = 0

    if parts:
        result.append(' '.join(parts))
    return result


# strips # to end-of-line from a GBNF source line, respecting string literals and char classes
def _strip_comment(line: str) -> str:
    i, in_str, in_cls, esc = 0, False, False, False
    while i < len(line):
        c = line[i]
        if esc:
            esc = False
        elif c == '\\':
            esc = True
        elif in_str:
            if c == '"': in_str = False
        elif in_cls:
            if c == ']': in_cls = False
        elif c == '"':
            in_str = True
        elif c == '[':
            in_cls = True
        elif c == '#':
            return line[:i]
        i += 1
    return line


# ── public API ───────────────────────────────────────────────────────────────


# parses a GBNF grammar string into:
#   terminals:   {T0: python_regex, T1: python_regex, …}  --- terminal labels mapped to their Python regexes
#   productions: {lhs: {(sym1, sym2, …), …}, …}           --- CFG productions; every symbol in a production
#                                                              tuple is either a key in terminals or in productions
# two-pass parse: first pass registers all LHS symbols so forward nonterminal references work correctly;
# second pass calls _parse_rhs for each rule, which may add auxiliary __grpN / __repN symbols
def parse_gbnf(gbnf_string: str) -> Tuple[Dict[str, str], Dict[str, Set[Tuple[str, ...]]]]:
    lines = _join_lines([s for line in gbnf_string.split('\n') if (s := _strip_comment(line).strip())])
    parsed = [line.split(' ::= ', 1) for line in lines]

    terminals: Dict[str, str] = {'""': '""'}  # '""' sentinel kept during construction; removed at end
    productions: Dict[str, Set[Tuple[str, ...]]] = {}
    aux_counter: List[int] = [0]

    # first pass: pre-register all rule LHS symbols so forward references in RHS are valid
    for lhs, _ in parsed:
        productions.setdefault(lhs.strip(), set())

    for lhs, rhs in parsed:
        _parse_rhs(lhs.strip(), rhs.strip(), terminals, productions, aux_counter)

    terminals.pop('""')
    terminals = {v: gbnf_to_regex(k) for k, v in terminals.items()}  # flip pattern=>label to label=>regex

    return terminals, productions
