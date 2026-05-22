
import torch
from typing import Optional, List
from cfgzip.utils import EquivalenceClassData, device_type


class MaskTranslator:
    """Expands a class-level grammar mask to the full token vocabulary.

    Takes the per-class bitmask produced by a grammar engine and broadcasts it
    to every token via ``token_classes``, then masks invalid tokens. Operates
    in-place on logit tensors so no extra allocation is needed at decode time.

    Args:
        eq_class_data: Precomputed equivalence class data from
            :func:`cfgzip.preprocess` or :meth:`EquivalenceClassData.load`.
        batch_size: Number of sequences being decoded in parallel.
        device: Torch device for mask tensors. Defaults to CUDA if available,
            otherwise CPU.
        logit_dtype: Dtype for the class mask. Defaults to ``float32`` on CPU
            and ``bfloat16`` on GPU.
    """

    class_representatives: List[bytes]

    def __init__(
            self,
            eq_class_data: EquivalenceClassData,
            batch_size: int = 1,
            device: device_type = None,
            logit_dtype: Optional[torch.dtype] = None
    ):
        if isinstance(device, torch.device): self.device = device
        else: self.device = torch.device((0 if torch.cuda.is_available() else 'cpu') if device is None else device)

        if logit_dtype is None: self.logit_dtype = torch.float32 if self.device.type == 'cpu' else torch.bfloat16
        else: self.logit_dtype = logit_dtype

        self.token_classes = eq_class_data.token_classes
        self.invalid_tokens = eq_class_data.invalid_tokens
        self.class_representatives = eq_class_data.class_representatives

        self.batch_size = batch_size
        self.n_tokens = self.token_classes.size(-1)
        self.n_classes = len(self.class_representatives)

        self.class_mask = torch.ones((self.batch_size, self.n_classes), device=self.device, dtype=self.logit_dtype)

    def mask_logits_inplace(
            self,
            logits: torch.Tensor,
            indices: Optional[List[int]] = None,
            mask_value: float = float('-inf')
    ) -> None:
        """Apply the current class mask to *logits* in-place and reset the mask.

        Tokens belonging to a disallowed class, and all unconditionally invalid
        tokens, are set to *mask_value*. The class mask is reset to all-ones
        after each call so the next grammar step starts from a clean state.

        Args:
            logits: Float tensor of shape ``(batch_size, vocab_size)`` to mask.
            indices: Batch indices to apply masking to. Defaults to all indices.
            mask_value: Value written to masked positions. Defaults to ``-inf``.
        """
        indices = list(range(self.batch_size)) if indices is None else indices
        logits[indices, :] = torch.where(self.class_mask[indices, self.token_classes] == 1.0, logits, mask_value)
        logits[indices, self.invalid_tokens] = mask_value
        self.class_mask[:, :] = 1.0
