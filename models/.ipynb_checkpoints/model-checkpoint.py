import torch
import torch.nn as nn
# from typing import Dict
from .data_containers import ModelOutput
from .file_handling import compress, decompress
from .part_HEIWA_Imp import HEIWA_Imp
from .part_HEIWA_Hybridizer import HEIWA_Hybridizer
from .part_TIM import TIM
from .part_SASA import ExposedResiduePrediction

class HEIWA3H(nn.Module):
    def __init__(self, hw_setting):
        super().__init__()
        self.parts = nn.ModuleDict()

        self.parts['TIM'] = TIM(
            in_dim=1536, out_dim=1, mid_dim=6,
        )
        self.parts["ExposedResiduePrediction"] = ExposedResiduePrediction(
            dim_prot=1536, 
            dim_intermediate=1024, 
            dim_out=1,
        )

        self.dim_v = 96
        self.num_solv_head = 4
        self.parts['HEIWA_Imp'] = HEIWA_Imp(
            le_cha_bias_action = -999, 
            num_prot_head=3, 
            z_dim_to_qk_dim_ratio=5, 
            num_solv_head=self.num_solv_head, 
            dim_qk=128,  
            dim_v=self.dim_v,
            num_rot_head=4, 
            rot_head_idx=[0, 1, 2, 3], 
            head_require_rot=[True, True, True, True], 
            head_label=['water', 'water', 'water', 'water'], 
            le_cha_bias_final = torch.tensor([0.2511, 0.2511, 0.2511, 0.2511], dtype=torch.float32, device=hw_setting.load_device), 
            dim_prot=1536, 
        )
        
        self.parts["HEIWA_Hybridizer"] = HEIWA_Hybridizer(
            in_dim=self.dim_v*self.num_solv_head*3, 
            v_compress=0.2, 
            expansion_factor=4, 
            out_dim=1536, 
        )
        
        for part in ["TIM", "ExposedResiduePrediction", "HEIWA_Imp", "HEIWA_Hybridizer"]: # "TIM", "ExposedResiduePrediction", 
            print(f"Loading Pretrained model weights for {part}")
            self.parts[part]._load_default(num_cpu=hw_setting.cpu, device=hw_setting.load_device)
            
    def move_to(self, device):
        self.parts['HEIWA_Imp'].le_cha_bias_final = self.parts['HEIWA_Imp'].le_cha_bias_final.to(device)
        self.to(device)
        
    def forward(self, x:torch.Tensor) -> ModelOutput:
        """Forward pass through the model."""
        bs, l, _ = x.size()

        tim_output = self.parts['TIM'](x)
        tim_gate = tim_output.y_pred
        
        ExposedResiduePrediction_output = self.parts["ExposedResiduePrediction"](x)
        sasa_gate = ExposedResiduePrediction_output.y_pred
        
        # Implicit
        heiwa_imp_output = self.parts['HEIWA_Imp'](x=x)
        water_modification_imp = heiwa_imp_output.y_pred

        # Hybydridizer
        hyd_output = self.parts["HEIWA_Hybridizer"](
            x_imp = water_modification_imp, 
            tim_gate = tim_gate, 
            sasa_gate = sasa_gate
        )
        water_modification = hyd_output.y_pred
        
        return x + water_modification
        
