# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-05-28

Initial release.

### Added
- `preprocess()`: offline preprocessing; computes equivalence classes for a
  `(grammar, tokenizer)` pair; returns `EquivalenceClassData` (implements `.save()` / `.load()` / `.to()`).
- `XgrammarProcessor`: XGrammar2 generation backend; a `transformers` `LogitsProcessor`, returned by
  `auto_pipeline()` and the `load_and_compile()` / `from_compiled()` static-CFG fast path (class methods).
- `MaskTranslator`: handles mapping between equivalence-class and token vocabularies
- `BaseProcessor`: abstract base defining the extension contract for future engines; super of `XgrammarProcessor`
- Grammars in GBNF; requires Python >= 3.10; optional `[xgrammar]` extra for the generation backend.

[Unreleased]: https://github.com/mjs227/cfgzip/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mjs227/cfgzip/releases/tag/v0.1.0
