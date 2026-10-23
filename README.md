# CFGzip

**Lossless token vocabulary compression for fast CFG-constrained decoding.**

CFGzip is an offline pre-computation technique that pairs with a constrained decoding engine (e.g. XGrammar2), to 
massively speed up the engine's inference compute time: generation with CFGzip+XGrammar2 is **up to 75x faster** than 
the SoTA XGrammar2 alone. CFGzip compression is also **lossless**: outputs are byte-identical to the unmodified grammar 
engine.

## How it works:

Within a given context-free grammar (CFG), many tokens are **interchangeable**: at any point where one is valid, so are 
the others, and vice versa. During the pre-compute phase, CFGzip detects these tokens and compresses them into a single 
*equivalence class*, keeping one representative per class. These equivalence classes are then cached — in memory or on 
disk (via `EquivalenceClassData.save()`) — to be used during generation.

At inference time, the grammar engine produces an allowed/disallowed mask over just the class representatives: this 
shrinks the per-step search space from 100-200k tokens to (typically) 1-3k equivalence class representatives. A 
`MaskTranslator` losslessly decompresses that class-level mask back to the full token vocabulary at each generation 
step.

See [our paper](https://arxiv.org/abs/2605.29986) for a description of the compression algorithm and correctness proofs.

## When to use it

CFGzip is intended for **static, large, complex grammars** that are reused across multiple requests: code generation, 
fixed schemas, structured DSLs. The offline compression step can take a few minutes, so it typically only pays off when 
one grammar serves many generations. CFGzip is **not** intended for dynamic, per-request schemas where you'd have to pay 
the precompute cost every time.

## Install

```bash
pip install cfgzip               # core: offline preprocessing (cfgzip-preprocess CLI)
pip install "cfgzip[xgrammar]"   # + the XGrammar generation backend
```

Requires Python ≥ 3.10. Installing compiles the Rust `cfgzip-preprocess` CLI from source and puts it on your `PATH`; 
this takes a minute or two. If no Rust toolchain (`cargo`) is found, one is downloaded automatically for the build. 
The build also needs a C compiler:

- **Linux:** `gcc`/`cc` (e.g. `build-essential` on Debian/Ubuntu), usually already installed.
- **macOS:** the Xcode Command Line Tools (`xcode-select --install`).
- **Windows:** the [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

## Quickstart

CFGzip has two phases: an **offline** step that computes and caches equivalence classes for a `(grammar, tokenizer)` 
pair, and an **online** step, where a third-party grammar engine does the heavy lifting — CFGzip just speeds things up.

### 1. Offline compression

Write the grammar to a file, e.g. `arithmetic.gbnf`:

```gbnf
root   ::= expr
expr   ::= term (("+" | "-") term)*
term   ::= factor (("*" | "/") factor)*
factor ::= [0-9]+ | "(" expr ")"
```

then run the `cfgzip-preprocess` CLI on it:

```bash
cfgzip-preprocess \
  --model-id gpt2 \
  --grammar-file arithmetic.gbnf \
  --output cfgzip_data/arithmetic \
  --num-threads 4
```

This writes the equivalence classes to `cfgzip_data/arithmetic` (which must be empty or not yet exist); load them with 
`EquivalenceClassData.load()`, or pass the directory straight to `XgrammarProcessor` (below). For gated HuggingFace 
models, set `HF_TOKEN` or pass `--hf-token`.

Useful options:

- `--model-id`: HuggingFace model id whose `tokenizer.json` should be used.
- `--grammar-file`: path to a GBNF grammar file.
- `--output`: empty or non-existent directory where artifacts will be written.
- `--start-symbol`: grammar start rule, default `root`.
- `--n-logits`: override the output logit vocabulary size.
- `--ignore-range START..END`: exclude token ids, useful for special tokens.
- `--skip-null-bytes`: put tokens containing `\x00` in singleton classes.
- `--skip-repeat-bytes`: put tokens containing a repeated decoded-byte run in singleton classes instead of processing 
  them through the grammar. This is a conservative preprocessing/runtime tradeoff for low-entropy tokenizer tokens such 
  as long runs of spaces.
- `--skip-repeat-min-run`: minimum repeated-byte run length for `--skip-repeat-bytes`, default `4`.
- `--num-threads`: worker thread count for token traversal.
- `--cache-dir`: HuggingFace asset cache directory.
- `--no-progress`: hide progress bars.

The pure-Python `preprocess()` function is still available, but is **deprecated** in favour of `cfgzip-preprocess`.

### 2. Online inference

`XgrammarProcessor` is a standard `transformers` `LogitsProcessor`, so it drops
straight into `model.generate`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from cfgzip import XgrammarProcessor

grammar = open('arithmetic.gbnf').read()
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

processor = XgrammarProcessor.auto_pipeline(
    'cfgzip_data/arithmetic', tokenizer, grammar, device=model.device
)

inputs = tokenizer('Calculator: ', return_tensors='pt').to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=16,
    logits_processor=LogitsProcessorList([processor]),
)
print(tokenizer.decode(out[0]))
```

### Static-CFG fast path

For the common case (one grammar, many batches) compile once and rebuild only the lightweight per-batch processor, 
skipping redundant disk I/O and grammar compilation:

```python
from cfgzip import XgrammarProcessor

mask_translator, compiled_grammar = XgrammarProcessor.load_and_compile(
    'cfgzip_data/arithmetic', tokenizer, grammar, device=model.device
)

for batch in batches:
    processor = XgrammarProcessor.from_compiled(mask_translator, compiled_grammar, tokenizer)
    output = model.generate(**batch, logits_processor=LogitsProcessorList([processor]))
    # ... do something
```

`auto_pipeline` is just `load_and_compile` + `from_compiled`; use the split form above whenever you generate more than 
once with the same grammar.

## Grammar format

Grammars are written in **GBNF** (GGML BNF) — the grammar notation used by 
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) and XGrammar2. The same grammar 
string drives both the offline `cfgzip-preprocess` step and online generation; see the linked guide for the full syntax.

Two CFGzip-specific constraints:

- CFGzip parses **core GBNF**; a few advanced constructs (e.g. `{m,n}` repetition counts) aren't supported yet.
- The start rule must be named `root` (or pass `--start-symbol` to `cfgzip-preprocess`), and the start symbol may **not** 
  appear in any rule body — keep the recursion on an inner non-terminal, as in the `root ::= expr` quickstart grammar 
  above.

## Scope & limitations

- **Engine backend:** v0.1.0 supports **XGrammar2 only**. `BaseProcessor` defines the contract for adding support for additional engines. We plan to support llguidance and transformers-cfg in later versions.
- **CFG notation:** similarly, `cfgzip-preprocess` only supports the GBNF grammar specification notation used by XGrammar2. Support for additional specification notations (e.g. Lark) is planned alongside support for decoding engines that use them.

## Public API

| Name | Description                                                                                        |
|---|----------------------------------------------------------------------------------------------------|
| `cfgzip-preprocess` | CLI; computes equivalence classes for a `(grammar, tokenizer)` pair (offline).                  |
| `preprocess` | **Deprecated** pure-Python equivalent of `cfgzip-preprocess`.                                      |
| `EquivalenceClassData` | The precomputed equivalence class data; `.save` / `.load` / `.to`.                                 |
| `XgrammarProcessor` | XGrammar wrapper and `LogitsProcessor`; `.auto_pipeline` / `.load_and_compile` / `.from_compiled`. |
| `MaskTranslator` | Expands a class-level mask back to the full token vocabulary in-place.                             |
| `BaseProcessor` | Abstract base / extension point for adding a new grammar engine.                                   |

## On AI usage

The main algorithm and functions were written by hand. We used Claude Code to port our research repository to a 
pip-installable module, including writing tests, docstrings, and portions of this README.

## Citation

If you use CFGzip in your research, please cite:

```bibtex
@article{sullivan2026accelerating,
  title   = {Accelerating Constrained Decoding with Token Space Compression},
  author  = {Sullivan, Michael and Koller, Alexander},
  journal = {arXiv preprint arXiv:2605.29986},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.29986}
}
```
