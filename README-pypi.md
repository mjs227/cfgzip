# CFGzip

**Lossless token vocabulary compression for fast CFG-constrained decoding.**

CFGzip is an offline pre-computation technique that pairs with a constrained decoding engine (e.g. XGrammar2) to
massively speed up the engine's inference: generation with CFGzip+XGrammar2 is **up to 7.5x faster** than SoTA
XGrammar2 alone, and **lossless** — outputs are byte-identical to the unmodified grammar engine.

```bash
pip install "cfgzip[xgrammar]"
```

Installing compiles the Rust `cfgzip-preprocess` CLI from source (Rust is downloaded automatically if missing) and
needs a C compiler: `gcc` on Linux, the Xcode Command Line Tools on macOS, or the Visual Studio C++ Build Tools on
Windows.

**Full documentation:** https://github.com/mjs227/cfgzip
