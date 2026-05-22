
import torch
from functools import wraps
from transformers import AutoTokenizer
from cfgzip.utils import EquivalenceClassData, device_type
from cfgzip.generation.mask_translator import MaskTranslator
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
from cfgzip.generation.logits_processors.base import BaseProcessor


if TYPE_CHECKING:
    import xgrammar as xgr
else:
    try:
        import xgrammar as xgr
    except ModuleNotFoundError:
        xgr = None


def xgr_check(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if xgr is None:
            raise ImportError('XGrammar is required for XgrammarProcessor; install cfgzip[xgrammar]')

        return func(*args, **kwargs)

    return wrapped



class XgrammarProcessor(BaseProcessor):
    """XGrammar-backed constrained decoding processor using cfgzip compression.

    Integrates a precomputed :class:`~cfgzip.EquivalenceClassData` with an
    XGrammar grammar matcher so the matcher operates over the compressed class
    vocabulary (typically ~10–100 classes) rather than the full token vocab.

    **Typical usage — static CFG (compile once, reuse across batches):**

    .. code-block:: python

        mask_translator, compiled_grammar = XgrammarProcessor.load_and_compile(
            filepath, tokenizer, grammar_str
        )
        # repeat for each batch:
        processor = XgrammarProcessor.from_compiled(
            mask_translator, compiled_grammar, tokenizer
        )

    **One-shot usage:**

    .. code-block:: python

        processor = XgrammarProcessor.auto_pipeline(
            filepath, tokenizer, grammar_str
        )

    Requires ``cfgzip[xgrammar]`` (``pip install cfgzip[xgrammar]``).
    """

    @xgr_check
    def __init__(
            self,
            eos_token_id: int,
            mask_translator: MaskTranslator,
            batch_matcher: "xgr.BatchGrammarMatcher",
            matchers: "List[xgr.GrammarMatcher]"
    ):
        super(XgrammarProcessor, self).__init__(eos_token_id, mask_translator)
        self.bitmask = xgr.allocate_token_bitmask(mask_translator.batch_size, mask_translator.n_classes)
        self.batch_matcher, self.matchers = batch_matcher, matchers

    def update_class_mask_inplace(self) -> None:
        self.batch_matcher.batch_fill_next_token_bitmask(self.matchers, self.bitmask, indices=self._live_batches)
        xgr.apply_token_bitmask_inplace(
            self.mask_translator.class_mask, self.bitmask.to(self.mask_translator.device), indices=self._live_batches
        )

    def accept_tokens(self, input_ids: torch.LongTensor) -> bool:
        return all(self.batch_matcher.batch_accept_token(
            [self.matchers[i] for i in self._live_batches],
            self.mask_translator.token_classes[input_ids[self._live_batches, -1]].flatten().tolist()
        ))

    @classmethod
    def auto_pipeline(
            cls,
            filepath: str,
            tokenizer: AutoTokenizer,
            grammar: str,
            batch_size: int = 1,
            device: device_type = None,
            logit_dtype: Optional[torch.dtype] = None,
            stop_token_ids: Optional[Union[List[int], int]] = None,
            max_threads: int = 8
    ) -> "XgrammarProcessor":
        """Load precomputed data, compile the grammar, and return a ready processor.

        Convenience wrapper around :meth:`load_and_compile` + :meth:`from_compiled`.
        Use this for one-off generation; for repeated generation with the same
        grammar prefer calling those two methods separately.

        Args:
            filepath: Directory written by :meth:`EquivalenceClassData.save`.
            tokenizer: HuggingFace tokenizer for the model being decoded.
            grammar: Grammar string (GBNF or CFG format).
            batch_size: Number of sequences to decode in parallel.
            device: Torch device for mask tensors. Defaults to CUDA if available.
            logit_dtype: Dtype for mask tensors. Defaults to bfloat16 on GPU.
            stop_token_ids: Token IDs treated as stop tokens. Defaults to
                ``[tokenizer.eos_token_id]``.
            max_threads: XGrammar thread pool size.
        """
        mask_translator, compiled_grammar = XgrammarProcessor.load_and_compile(
            filepath, tokenizer, grammar, batch_size, device, logit_dtype, stop_token_ids, max_threads
        )

        return XgrammarProcessor.from_compiled(mask_translator, compiled_grammar, tokenizer, batch_size, max_threads)

    @classmethod
    @xgr_check
    def load_and_compile(
            cls,
            filepath: str,
            tokenizer: AutoTokenizer,
            grammar: str,
            batch_size: int = 1,
            device: device_type = None,
            logit_dtype: Optional[torch.dtype] = None,
            stop_token_ids: Optional[Union[List[int], int]] = None,
            max_threads: int = 8
    ) -> "Tuple[MaskTranslator, xgr.CompiledGrammar]":
        """Load precomputed data and compile the grammar for repeated use.

        Returns a ``(MaskTranslator, CompiledGrammar)`` pair that can be passed
        to :meth:`from_compiled` once per batch, avoiding redundant compilation
        and disk I/O when the same grammar is used across many batches.

        Args:
            filepath: Directory written by :meth:`EquivalenceClassData.save`.
            tokenizer: HuggingFace tokenizer for the model being decoded.
            grammar: Grammar string (GBNF or CFG format).
            batch_size: Number of sequences to decode in parallel.
            device: Torch device for mask tensors. Defaults to CUDA if available.
            logit_dtype: Dtype for mask tensors. Defaults to bfloat16 on GPU.
            stop_token_ids: Token IDs treated as stop tokens. Defaults to
                ``[tokenizer.eos_token_id]``.
            max_threads: XGrammar thread pool size.

        Returns:
            ``(mask_translator, compiled_grammar)`` — pass both to
            :meth:`from_compiled` to create a processor.
        """
        eq_data = EquivalenceClassData.load(filepath, device=device)
        mask_translator = MaskTranslator(eq_data, batch_size=batch_size, device=device, logit_dtype=logit_dtype)

        if stop_token_ids is None: stop_token_ids = [tokenizer.eos_token_id]
        stop_token_ids = sorted(set(eq_data.token_classes[stop_token_ids].tolist()))

        tokenizer_info = xgr.TokenizerInfo(eq_data.class_representatives, stop_token_ids=stop_token_ids)
        compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=max_threads)
        compiled_grammar = compiler.compile_grammar(grammar)

        return mask_translator, compiled_grammar

    @classmethod
    @xgr_check
    def from_compiled(
            cls,
            mask_translator: MaskTranslator,
            compiled_grammar: "xgr.CompiledGrammar",
            tokenizer: AutoTokenizer,
            batch_size: int = 1,
            max_threads: int = 8
    ) -> "XgrammarProcessor":
        """Create a processor from an already-compiled grammar.

        Pair with :meth:`load_and_compile` for the static-CFG fast path: compile
        once, then call this once per batch.

        Args:
            mask_translator: From :meth:`load_and_compile`.
            compiled_grammar: From :meth:`load_and_compile`.
            tokenizer: HuggingFace tokenizer for the model being decoded.
            batch_size: Number of sequences to decode in parallel.
            max_threads: XGrammar thread pool size.
        """
        matchers = [xgr.GrammarMatcher(compiled_grammar) for _ in range(batch_size)]
        batch_matcher = xgr.BatchGrammarMatcher(max_threads=max_threads)

        return XgrammarProcessor(tokenizer.eos_token_id, mask_translator, batch_matcher, matchers)
