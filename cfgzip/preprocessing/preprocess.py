
import re
import torch
from transformers import PreTrainedTokenizer
from cfgzip.utils import EquivalenceClassData
from typing import Tuple, Optional, Callable, List, Dict
from cfgzip.preprocessing.parse_cfg_str import parse_cfg_str
from cfgzip.preprocessing.normalize_cfg import normalize_cfg
from cfgzip.preprocessing.compute_token_classes import compute_token_classes


def gpt2_byte_decoder() -> Dict[str, int]:
    # invert GPT-2 bytes_to_unicode map to get mapping char => byte.
    out, n = {}, 0

    for b in range(256):
        # if b
        if b < 33 or 127 <= b <= 160 or b == 173:
            out.update({chr(n + 256): b})
            n += 1
        else:
            out.update({chr(b): b})

    return out


# extract raw token bytes
def token_to_bytes(token: str, byte_decoder: Dict[str, int]) -> Tuple[int, ...]:
    if m := re.fullmatch(r'<0x([0-9A-Fa-f]{2})>', token):  # byte-fallback tokens: <0x0A>, <0x1B>, ...
        return (int(m.group(1), 16),)

    return tuple(byte_decoder[c] for c in token)


def preprocess(
        grammar_str: str,
        tokenizer: PreTrainedTokenizer,
        start_symbol: str = 'root',
        skip_compute_tokens: Optional[Callable[[Tuple[int, ...], int], bool]] = None,
        ignore_tokens: Optional[Callable[[Tuple[int, ...], int], bool]] = None,
        n_logits: Optional[int] = None,
        use_tqdm: bool = False,
        num_workers: int = 1
) -> EquivalenceClassData:
    """Compute token equivalence classes for a (grammar, tokenizer) pair.

    This is the offline preprocessing step. Save the result with
    :meth:`EquivalenceClassData.save` and reload it at generation time with
    :meth:`EquivalenceClassData.load`.

    Args:
        grammar_str: Grammar in GBNF or CFG string format.
        tokenizer: HuggingFace tokenizer for the model being constrained.
        start_symbol: Name of the grammar's start non-terminal. Defaults to
            ``'root'``.
        skip_compute_tokens: Optional callback ``(bytes, token_id) -> bool``.
            If it returns ``True`` for a token, displacement computation for
            that token is skipped entirely (token is treated as having no valid
            continuations). Receives raw byte tuple, not a string.
        ignore_tokens: Optional callback ``(bytes, token_id) -> bool``.
            If it returns ``True``, the token is excluded from the vocabulary
            before equivalence class computation (e.g. special tokens).
            Receives raw byte tuple, not a string.
        n_logits: Override the vocab size (number of logit positions). Inferred
            from the tokenizer if not given.
        use_tqdm: Show a tqdm progress bar over token displacement computation.
        num_workers: Number of worker processes for parallel computation.

    Returns:
        :class:`EquivalenceClassData` with the computed equivalence classes.
    """
    cfg, lex_grammar, preterminals, terminal_labels = parse_cfg_str(grammar_str, start_symbol=start_symbol)
    normed_cfg = normalize_cfg(cfg, lex_grammar, terminal_labels)

    if not (hasattr(tokenizer, 'eos_token_id') and isinstance(tokenizer.eos_token_id, int)):
        raise ValueError(
            "tokenizer must have an integer eos_token_id; "
            f"got tokenizer.eos_token_id = {getattr(tokenizer, 'eos_token_id', '<missing>')!r}"
        )
    byte_decoder = getattr(tokenizer, 'byte_decoder', gpt2_byte_decoder())  # fast tokenizers don't expose byte_decoder
    tokens = [
        (token_to_bytes(tokenizer.convert_ids_to_tokens(v), byte_decoder), v) for v in tokenizer.get_vocab().values()
    ]

    token_classes, invalid_tokens, class_representatives = compute_token_classes(
        normed_cfg, preterminals, tokens, tokenizer.eos_token_id, skip_compute_tokens,
        ignore_tokens, n_logits, use_tqdm, num_workers
    )

    return EquivalenceClassData(
        torch.tensor(token_classes, dtype=torch.int32),
        torch.tensor(invalid_tokens, dtype=torch.int32),
        class_representatives
    )
