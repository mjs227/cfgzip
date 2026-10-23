"""Rust ``cfgzip-preprocess`` CLI: its artifacts load in Python and mask byte-identically to raw XGrammar2."""

import shutil
import subprocess

import pytest

from cfgzip.preprocessing.preprocess import preprocess
from cfgzip.utils import EquivalenceClassData
from test_e2e import (
    MULTI_STEP_CASES, GRAMMAR_ALPHA, tokenizer, raw_vocab, _make_cfgzip_proc, _make_raw_xgr, _compare_steps,
)

CLI = shutil.which("cfgzip-preprocess")
pytestmark = pytest.mark.skipif(CLI is None, reason="cfgzip-preprocess binary not on PATH")


def _rust_preprocess(grammar: str, tmp_path) -> EquivalenceClassData:
    grammar_file, out_dir = tmp_path / "grammar.gbnf", tmp_path / "out"
    grammar_file.write_text(grammar)
    subprocess.run(
        [CLI, "--model-id", "gpt2", "--grammar-file", str(grammar_file), "--output", str(out_dir), "--no-progress"],
        check=True,
    )
    return EquivalenceClassData.load(str(out_dir))


@pytest.mark.parametrize("grammar,n_steps,_id", MULTI_STEP_CASES, ids=[c[2] for c in MULTI_STEP_CASES])
def test_rust_multi_step_byte_identical(grammar, n_steps, _id, tokenizer, raw_vocab, tmp_path):
    """Rust-preprocessed artifacts and raw XGrammar produce byte-identical masks at every decode step."""
    eq = _rust_preprocess(grammar, tmp_path)
    proc, _ = _make_cfgzip_proc(grammar, tokenizer, eq)
    raw_batch, raw_matcher = _make_raw_xgr(grammar, raw_vocab, tokenizer.eos_token_id)
    _compare_steps(proc, raw_batch, raw_matcher, n_steps=n_steps)


def test_rust_artifacts_match_python_format(tokenizer, tmp_path):
    """Rust artifacts have the same tensor shapes/dtypes and representative type as the Python preprocessor.

    Class *counts* are deliberately not compared: the Rust preprocessor may merge language-equivalent
    terminal-regex states and so produce fewer classes.
    """
    rust = _rust_preprocess(GRAMMAR_ALPHA, tmp_path)
    py = preprocess(GRAMMAR_ALPHA, tokenizer, num_workers=1)

    assert rust.token_classes.shape == py.token_classes.shape
    assert rust.token_classes.dtype == py.token_classes.dtype
    assert rust.invalid_tokens.dtype == py.invalid_tokens.dtype
    assert all(isinstance(rep, type(py.class_representatives[0])) for rep in rust.class_representatives)
