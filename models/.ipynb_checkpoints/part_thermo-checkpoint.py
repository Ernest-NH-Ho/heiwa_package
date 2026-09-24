import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .file_handling import compress, decompress
from .basic_layers import Swish
from .data_containers import ModelOutput
from ..utils import asset_path

class ThermoPred(nn.Module):
    def __init__(self, hw_setting):
        super().__init__()
        self.intermediate_dim = 96
        self.swish1 = Swish(in_dim=1536, out_dim=96)
        self.swish2 = Swish(in_dim=96, out_dim=1)
        self._load_default(num_cpu=hw_setting.cpu, device=hw_setting.load_device)
        

    def _load_default(self, num_cpu, device):
        path = asset_path.joinpath(f"Thermo_epoch_final_model.pth")
        decompress(f"{path}", num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location=device, weights_only=True)
        incompatible = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", num_cpu=num_cpu)
    
    def move_to(self, device):
        self.to(device)
        
    def forward(self, model_input:Dict[str, torch.Tensor], water_modification:torch.Tensor|float=0) -> torch.Tensor:
        bs, l, _ = model_input["X"].size()
        x = model_input["X"]+water_modification
        middle_layer = self.swish1(x)
        
        final_layer = self.swish2(middle_layer).reshape(bs, l)
        if "M" in model_input:
            final_layer = final_layer * model_input["M"]
        
        # if target.lower()=="rmsf":
        # intermediate_output = final_layer
        # return ModelOutput(y_pred=final_layer, intermediate_output=final_layer)
            
        y_pred = final_layer.sum(dim=-1) / model_input["L"]
        y_pred = F.sigmoid(y_pred)
        return ModelOutput(y_pred=y_pred, intermediate_output=final_layer)
