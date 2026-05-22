"""Byte-identical masking: cfgzip == raw XGrammar2, across multiple decode steps."""

import pytest
import torch
import xgrammar as xgr
from transformers import AutoTokenizer

from cfgzip.preprocessing.preprocess import preprocess, gpt2_byte_decoder, token_to_bytes
from cfgzip.generation.mask_translator import MaskTranslator
from cfgzip.generation.logits_processors.xgrammar_processor import XgrammarProcessor
from cfgzip.utils import EquivalenceClassData

GRAMMAR_ALPHA = "root ::= [a-z]+"

# Multi-rule CFG: key ":" value (letter-phase → digit-phase, proper non-terminals)
_GRAMMAR_KV = """\
root ::= key ":" value
key ::= [a-z]+
value ::= [0-9]+"""

# ── GNF-stressing grammars: each drives a distinct normalization / parse path. ──
# Invariant (required by parse_cfg_str, which renames only the LHS `root` → `S`):
# `root` is the LHS of exactly one entry rule and never appears in any rule body —
# so recursion lives on an inner non-terminal.

# Local left recursion (expr ::= expr …) → LR[expr] removal + unit rules.
_GRAMMAR_LEFT_REC = """
root ::= expr
expr ::= expr "+" term | term
term ::= [0-9]
"""

# Indirect (mutual) left recursion a→b→a; language y(zx)* → global LR-removal pass.
_GRAMMAR_INDIRECT_LR = """
root ::= a
a ::= b "x" | "y"
b ::= a "z"
"""

# Self-embedding (Dyck-like) recursion: non-terminal in the middle of the RHS.
_GRAMMAR_NESTED = """
root ::= parens
parens ::= "(" parens ")" | "x"
"""

# CFG-level epsilon (sign → ε) + right recursion (num) → remove_epsilon_productions.
_GRAMMAR_OPT_SIGN = """
root ::= sign num
sign ::= "-" | ""
num ::= [0-9] num | [0-9]
"""

# Group + star aux rules (__grp / __rep) → CFG-level ε.
_GRAMMAR_COMMA_LIST = """
root ::= item ("," item)*
item ::= [a-z]
"""

# Duplicate non-terminals a ≡ b → the dedup/merge loop in cfg_to_gnf.
_GRAMMAR_DUP_MERGE = """
root ::= a b
a ::= "x" c | "y" c
b ::= "x" c | "y" c
c ::= [0-9] c | [0-9]
"""

# Unit-rule chain word→a→b (remove_unary_rules) + right recursion.
_GRAMMAR_UNIT_CHAIN = """
root ::= items
items ::= word items | word
word ::= a
a ::= b
b ::= "ab" | "cd"
"""

# (grammar, n_steps, id): grammars with different structural patterns
MULTI_STEP_CASES = [
    ("root ::= [a-z]+",        6, "alpha"),            # simple repetition
    ("root ::= [0-9]+",        6, "digits"),            # different token classes
    ("root ::= [a-z] [0-9]+",  4, "alpha-digit"),       # hard single-step state transition
    ("root ::= [a-z]+ [0-9]+", 5, "alpha-plus-digit"),  # flexible transition mid-sequence
    (_GRAMMAR_KV,              5, "kv-json-like"),       # multi-rule CFG, letter→colon→digit
    # ── GNF / pipeline path coverage (root never on a RHS) ──
    (_GRAMMAR_LEFT_REC,        4, "left-rec"),          # local left recursion (LR[expr])
    (_GRAMMAR_INDIRECT_LR,     4, "indirect-lr"),       # global / indirect left recursion
    (_GRAMMAR_NESTED,          4, "nested-parens"),     # self-embedding (Dyck-like) recursion
    (_GRAMMAR_OPT_SIGN,        4, "optional-sign"),     # CFG-level epsilon removal
    (_GRAMMAR_COMMA_LIST,      4, "comma-list"),        # group + star aux rules
    (_GRAMMAR_DUP_MERGE,       4, "dup-merge"),         # duplicate non-terminal merge
    ('root ::= ("ab" | "cd")+', 4, "alt-terminal"),     # multi-char / branching terminal NFA
    (_GRAMMAR_UNIT_CHAIN,      4, "unit-chain"),         # unit-rule removal chain
]


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def raw_vocab(tokenizer):
    byte_decoder = getattr(tokenizer, "byte_decoder", gpt2_byte_decoder())
    vocab = tokenizer.get_vocab()
    n = tokenizer.vocab_size
    result: list[bytes] = [b""] * n
    for tok_str, tok_id in vocab.items():
        if tok_id < n:
            result[tok_id] = bytes(token_to_bytes(tok_str, byte_decoder))
    return result


@pytest.fixture(scope="module")
def eq_alpha(tokenizer):
    return preprocess(GRAMMAR_ALPHA, tokenizer, num_workers=1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cfgzip_proc(grammar: str, tokenizer, eq_data: EquivalenceClassData):
    mt = MaskTranslator(eq_data, batch_size=1)
    stop_cls = sorted(set(eq_data.token_classes[[tokenizer.eos_token_id]].tolist()))
    tok_info = xgr.TokenizerInfo(eq_data.class_representatives, stop_token_ids=stop_cls)
    compiled = xgr.GrammarCompiler(tok_info, max_threads=1).compile_grammar(grammar)
    proc = XgrammarProcessor(
        tokenizer.eos_token_id, mt,
        xgr.BatchGrammarMatcher(max_threads=1),
        [xgr.GrammarMatcher(compiled)],
    )
    return proc, mt


def _make_raw_xgr(grammar: str, raw_vocab: list[bytes], eos_token_id: int):
    tok_info = xgr.TokenizerInfo(raw_vocab, stop_token_ids=[eos_token_id])
    compiled = xgr.GrammarCompiler(tok_info, max_threads=1).compile_grammar(grammar)
    batch = xgr.BatchGrammarMatcher(max_threads=1)
    matcher = xgr.GrammarMatcher(compiled)
    return batch, matcher


def _assert_masks_identical(logits_c: torch.Tensor, logits_r: torch.Tensor, label: str):
    masked_c = logits_c.cpu().float()[0] == float("-inf")
    masked_r = logits_r[0] == float("-inf")
    n = (masked_c != masked_r).sum().item()
    if n:
        only_c = (masked_c & ~masked_r).nonzero().flatten().tolist()[:10]
        only_r = (masked_r & ~masked_c).nonzero().flatten().tolist()[:10]
        pytest.fail(f"{label}: {n} token(s) differ — cfgzip-only={only_c}, raw-only={only_r}")


def _compare_steps(proc: XgrammarProcessor, raw_batch, raw_matcher, n_steps: int):
    """Assert cfgzip and raw XGrammar produce byte-identical masks for n_steps steps.

    Advances both paths by accepting the first allowed non-EOS token at each step.
    """
    mt = proc.mask_translator
    n_tokens = mt.n_tokens
    input_ids = torch.zeros(1, 1, dtype=torch.long)  # dummy — step 0 ignores this

    for step in range(n_steps):
        logits_c = torch.zeros(1, n_tokens, device=mt.class_mask.device, dtype=mt.logit_dtype)
        proc(input_ids, logits_c)

        logits_r = torch.zeros(1, n_tokens)
        bitmask = xgr.allocate_token_bitmask(1, n_tokens)
        raw_batch.batch_fill_next_token_bitmask([raw_matcher], bitmask)
        xgr.apply_token_bitmask_inplace(logits_r, bitmask)

        _assert_masks_identical(logits_c, logits_r, f"step {step}")

        allowed = (logits_c.cpu().float()[0] != float("-inf")).nonzero().flatten().tolist()
        non_eos = [t for t in allowed if t != proc.eos_token_id]
        assert non_eos, f"step {step}: grammar terminated before {n_steps} steps"
        chosen = non_eos[0]

        raw_batch.batch_accept_token([raw_matcher], [chosen])
        input_ids = torch.tensor([[chosen]], dtype=torch.long)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_class_count(eq_alpha):
    """[a-z]+ on gpt2 yields exactly 3 equivalence classes (regression gate)."""
    assert len(eq_alpha.class_representatives) == 3


@pytest.mark.parametrize("grammar,n_steps,_id", MULTI_STEP_CASES, ids=[c[2] for c in MULTI_STEP_CASES])
def test_multi_step_byte_identical(grammar, n_steps, _id, tokenizer, raw_vocab):
    """cfgzip and raw XGrammar produce byte-identical masks at every decode step."""
    eq = preprocess(grammar, tokenizer, num_workers=1)
    proc, _ = _make_cfgzip_proc(grammar, tokenizer, eq)
    raw_batch, raw_matcher = _make_raw_xgr(grammar, raw_vocab, tokenizer.eos_token_id)
    _compare_steps(proc, raw_batch, raw_matcher, n_steps=n_steps)


def test_save_load_roundtrip(eq_alpha, tmp_path):
    """save() then load() preserves token_classes, invalid_tokens, class_representatives."""
    eq_alpha.save(str(tmp_path / "grammar"))
    loaded = EquivalenceClassData.load(str(tmp_path / "grammar"))

    assert torch.equal(eq_alpha.token_classes, loaded.token_classes)
    assert torch.equal(eq_alpha.invalid_tokens, loaded.invalid_tokens)
    assert eq_alpha.class_representatives == loaded.class_representatives
