
import torch
from abc import ABC, abstractmethod
from transformers import LogitsProcessor
from cfgzip.generation.mask_translator import MaskTranslator


class BaseProcessor(LogitsProcessor, ABC):
    """Abstract base for cfgzip-backed constrained decoding processors.

    Subclasses integrate a specific grammar engine with :class:`~cfgzip.MaskTranslator`
    to perform constrained decoding. Implement the two abstract methods to wire
    in a new engine; everything else (batch tracking, EOS handling, logit masking)
    is handled here.

    To add a new engine backend, subclass and implement:
        - :meth:`update_class_mask_inplace` — fill ``mask_translator.class_mask``
          with the allowed-class bitmask for the current decoding step.
        - :meth:`accept_tokens` — advance the grammar matcher state by accepting
          the tokens chosen at the previous step; return ``False`` if any token
          is rejected (indicates a generation error).
    """

    def __init__(self, eos_token_id: int, mask_translator: MaskTranslator):
        self.mask_translator, self.eos_token_id = mask_translator, eos_token_id
        self._live_batches = list(range(self.mask_translator.batch_size))
        self._step_gt_0 = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._step_gt_0:  # update MaskTranslator with tokens from prev. step
            self._live_batches = [i for i in self._live_batches if input_ids[i, -1] != self.eos_token_id]

            if self._live_batches:
                assert self.accept_tokens(input_ids)
            else:
                return scores
        else:
            self._step_gt_0 = True

        self.update_class_mask_inplace()
        self.mask_translator.mask_logits_inplace(scores, indices=self._live_batches)

        return scores

    def accept_prefix(self, input_ids: torch.LongTensor):
        for i in range(input_ids.size(-1)):
            if not self.accept_tokens(input_ids[:, i].unsqueeze(0)):
                return False

        return True

    @abstractmethod
    def update_class_mask_inplace(self) -> None:
        """Fill ``self.mask_translator.class_mask`` with the allowed-class bitmask.

        Called once per decode step before logit masking. Must update the mask
        in-place for all indices in ``self._live_batches``.
        """
        pass

    @abstractmethod
    def accept_tokens(self, input_ids: torch.LongTensor) -> bool:
        """Advance the grammar matcher by accepting the last token of each live sequence.

        Called after each decode step with the full ``input_ids`` tensor. Must
        return ``False`` if any live sequence produced a token the grammar rejects.
        """
        pass
