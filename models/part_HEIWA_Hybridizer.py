import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from heiwa.models.basic_layers import Swish, SwiGLU_Simplified
from heiwa.models.file_handling import load_weights
from heiwa.models.data_containers import ModelOutput
from heiwa.utils import asset_path

class HEIWA_Hybridizer(nn.Module):
    def __init__(self, in_dim, v_compress, expansion_factor, out_dim, dropout=None):
        super().__init__()
        mid_dim = round(in_dim*v_compress)
        self.Wo_w = Swish(in_dim=in_dim, out_dim=mid_dim, dropout_value=0)
        self.out_proj = SwiGLU_Simplified(in_dim=mid_dim, out_dim=out_dim, dropout_value=0, 
                                          expansion_factor=expansion_factor)
        self.out_proj.pretrained = False

    def _load_default(self, num_cpu, device):
        path = asset_path.joinpath(f"HEIWA_Hybridizer_epoch_final_model.pth")
        state_dict = load_weights(path, device)
        incompatible = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        
    def forward(self, x_imp=None, tim_gate=0, sasa_gate=1):
        """ RMSNorm should be done by the part that produces the x_imp and x_exp """
        bs, l, _ = x_imp.size()

        water_modification = self.Wo_w(x_imp)

        if isinstance(tim_gate, torch.Tensor):
            tim_gate = tim_gate.reshape(bs, l, 1)
        if isinstance(sasa_gate, torch.Tensor):
            sasa_gate = sasa_gate.reshape(bs, l, 1)
            
        water_modification = self.out_proj(water_modification) * (1-tim_gate) * sasa_gate
        return ModelOutput(y_pred=water_modification)