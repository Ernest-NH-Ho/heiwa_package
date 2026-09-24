from dataclasses import dataclass
import torch

        
@dataclass
class ModelOutput:
    y_pred: torch.Tensor
    intermediate_output: torch.Tensor|None=None
