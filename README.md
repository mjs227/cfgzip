# cfgzip

**Token-vocabulary compression for fast, lossless constrained LLM decoding.**

cfgzip groups a model's tokens into *equivalence classes* that are interchangeable
with respect to a context-free grammar, so the grammar engine masks over a handful
of class representatives (typically ~10–100) instead of the full 100k–200k-token
vocabulary. The per-step output is then expanded back to the token level — with no
change to what the model is allowed to generate.

- **>20× less constrained-decoding overhead** (up to ~2 orders of magnitude).
- **Up to ~10× faster** end-to-end vs. SoTA XGrammar2.
- **Lossless** — byte-identical to the unmodified grammar engine.

> The first two are *different* metrics: overhead reduction (how much cheaper the
> masking step is) is not the same as absolute end-to-end speedup. Both figures
> are from the paper; see [Citation](#citation).

## When to use it

cfgzip shines on **static, large, complex grammars** that you reuse across many
requests — code generation, fixed schemas, structured DSLs. The class computation
is an **offline** step (seconds to a few minutes per grammar, cached to disk), so
it pays off when one grammar serves many generations. It is **not** intended for
dynamic, per-request schemas where you'd pay the precompute cost every time.

## Install

```bash
pip install cfgzip               # core: offline preprocessing
pip install "cfgzip[xgrammar]"   # + the XGrammar generation backend
```

Requires Python ≥ 3.10. (Not yet on PyPI — until the first release, install from
source with `pip install ".[xgrammar]"`.)

## Quickstart

cfgzip has two phases: an **offline** step that computes and caches equivalence
classes for a `(grammar, tokenizer)` pair, and an **online** step that uses them
to constrain generation.

### 1. Offline — precompute once, save to disk

```python
from transformers import AutoTokenizer
from cfgzip import preprocess

tokenizer = AutoTokenizer.from_pretrained("gpt2")
grammar = "root ::= [a-z]+"          # GBNF / CFG string

eq = preprocess(grammar, tokenizer, num_workers=4)
eq.save("grammars/lowercase")
```

### 2. Online — load and generate

`XgrammarProcessor` is a standard `transformers` `LogitsProcessor`, so it drops
straight into `model.generate`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from cfgzip import XgrammarProcessor

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

processor = XgrammarProcessor.auto_pipeline(
    "grammars/lowercase", tokenizer, grammar, device=model.device
)

inputs = tokenizer("the password is: ", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=16,
    logits_processor=LogitsProcessorList([processor]),
)
print(tokenizer.decode(out[0]))
```

> Pass `device=model.device` so the mask tensors live on the same device as the
> model. Left unset, cfgzip defaults to CUDA-if-available, which mismatches a
> CPU model.

### Static-CFG fast path

For the common case — one grammar, many batches — compile once and rebuild only
the lightweight per-batch processor, skipping redundant disk I/O and grammar
compilation:

```python
from cfgzip import XgrammarProcessor

mt, compiled = XgrammarProcessor.load_and_compile(
    "grammars/lowercase", tokenizer, grammar, device=model.device
)

for batch in batches:
    processor = XgrammarProcessor.from_compiled(mt, compiled, tokenizer)
    model.generate(**batch, logits_processor=LogitsProcessorList([processor]))
```

`auto_pipeline` is just `load_and_compile` + `from_compiled`; use the split form
whenever you generate more than once with the same grammar.

## How it works

Within a given grammar, many tokens are **interchangeable**: at any point where one
is valid, so are the others, and vice versa. cfgzip detects these tokens and
collapses them into a single *equivalence class*, keeping one representative per
class. At decode time the grammar engine produces an allowed/disallowed mask over
just the class representatives, and a `MaskTranslator` expands that class-level
mask back to the full token vocabulary in-place — once per step.

Computing the classes is the offline cost (cached via `EquivalenceClassData.save`);
generation is online and operates entirely over the compressed class space. Because
the expansion is exact, the set of allowed tokens at every step is identical to the
uncompressed engine's — the compression is provably lossless. See the paper for the
displacement-based algorithm and the correctness proof.

## Scope & limitations

- **Backend:** v1 supports **XGrammar only**. Other engines are a planned
  extension — `BaseProcessor` defines the contract for adding one.
- **Offline precompute** is per `(grammar, tokenizer)` and costs seconds to
  minutes; cfgzip is built for static grammars reused across many requests, not
  dynamic per-request schemas.
- The lossless guarantee is exercised by a **byte-identical** end-to-end test
  (`tests/test_e2e.py`): cfgzip's per-step mask matches raw XGrammar exactly.

## Public API

All names below are importable from the top-level `cfgzip` package; the package
ships type hints (`py.typed`), and each item is fully documented in its docstring.

| Name | What it does |
|---|---|
| `preprocess` | Offline: compute equivalence classes for a `(grammar, tokenizer)` pair. |
| `EquivalenceClassData` | The precomputed data, with `.save` / `.load` / `.to`. |
| `XgrammarProcessor` | XGrammar-backed `LogitsProcessor`; `.auto_pipeline` / `.load_and_compile` / `.from_compiled`. |
| `MaskTranslator` | Expands a class-level mask back to the full token vocabulary in-place. |
| `BaseProcessor` | Abstract base / extension point for adding a new grammar engine. |

## Citation

Paper under review — citation coming soon.
