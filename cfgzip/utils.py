
import os
import torch
import pickle
from dataclasses import dataclass
from typing import List, Optional, Union


device_type = Optional[Union[str, int, torch.device]]


@dataclass
class EquivalenceClassData:
    """Precomputed token equivalence class data for a (grammar, tokenizer) pair.

    Maps each token to an equivalence class whose members are interchangeable
    with respect to a CFG, and records one representative byte sequence per
    class. Produced by :func:`cfgzip.preprocess` and consumed by
    :class:`cfgzip.MaskTranslator`.

    Attributes:
        token_classes: 1-D int32 tensor of length ``vocab_size``; entry ``i``
            is the class ID of token ``i``.
        invalid_tokens: 1-D int32 tensor of token IDs that are invalid in
            every possible context and are always masked out.
        class_representatives: List of length ``n_classes``; entry ``k`` is
            the raw byte sequence of the representative token for class ``k``.

    On-disk layout (written by :meth:`save`, read by :meth:`load`):
        ``tc.pt``  — ``token_classes`` tensor (torch.save format)
        ``inv.pt`` — ``invalid_tokens`` tensor (torch.save format)
        ``cr.pkl`` — ``class_representatives`` list (pickle, trusted local files per D9)
    """

    # token_classes[i] = class ID of token i
    token_classes: torch.Tensor
    # set of tokens that are invalid in all possible contexts (saved as tensor for fast indexing)
    invalid_tokens: torch.Tensor
    # class_representatives[k] = byte encoding of representative of equivalence class k
    class_representatives: List[bytes]

    def to(self, device: device_type) -> None:
        """Move the tensor fields to *device* in-place (``class_representatives`` is unaffected)."""
        self.token_classes = self.token_classes.to(device=device)
        self.invalid_tokens = self.invalid_tokens.to(device=device)

    def save(self, filepath: str) -> None:
        """Save to *filepath* (must be an empty directory or not yet exist).

        Creates the directory if absent. Raises ``ValueError`` if the path
        exists and is not an empty directory, to prevent silent overwrites.
        """
        filepath = os.path.abspath(filepath.rstrip('/'))

        if os.path.exists(filepath):
            if not os.path.isdir(filepath):
                raise ValueError(f"filepath exists and is not a directory: {filepath!r}")
            if os.listdir(filepath):
                raise ValueError(f"filepath directory is not empty: {filepath!r}")
        else:
            os.makedirs(filepath)

        torch.save(self.token_classes, f'{filepath}/tc.pt')
        torch.save(self.invalid_tokens, f'{filepath}/inv.pt')
        with open(f'{filepath}/cr.pkl', 'wb') as f: pickle.dump(self.class_representatives, f)

    @classmethod
    def load(cls, filepath: str, device: device_type = None) -> "EquivalenceClassData":
        """Load from a directory previously written by :meth:`save`.

        Args:
            filepath: Path to the saved grammar directory.
            device: Torch device for the loaded tensors. Defaults to CPU.

        Returns:
            :class:`EquivalenceClassData` loaded from *filepath*.
        """
        filepath = os.path.abspath(filepath.rstrip('/'))
        device = torch.device('cpu') if device is None else device

        token_classes = torch.load(f'{filepath}/tc.pt', map_location=device)
        invalid_tokens = torch.load(f'{filepath}/inv.pt', map_location=device)
        with open(f'{filepath}/cr.pkl', 'rb') as f: class_representatives = pickle.load(f)

        return EquivalenceClassData(token_classes, invalid_tokens, class_representatives)
