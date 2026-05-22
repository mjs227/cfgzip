
import torch
from cfgzip.generation.logits_processors.base import BaseProcessor


class TransformersCFGProcessor(BaseProcessor):
    def __init__(self):
        raise NotImplementedError("TransformersCFGProcessor is not implemented in v1 (D2)")

    def accept_tokens(self, input_ids: torch.LongTensor) -> bool:
        pass

    def update_class_mask_inplace(self) -> None:
        pass
