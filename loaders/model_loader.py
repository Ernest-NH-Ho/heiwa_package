import torch
from heiwa.models.model import HEIWA3H
from heiwa.models.orthogonalizer import HEIWA_Orthogonalizer
from heiwa.models.part_thermo import ThermoPred
from heiwa.utils import asset_path

# def load_esm3(device="cuda"):
#     with torch.no_grad():
#         return torch.load(
#             asset_path.joinpath("esm3.pth"), 
#             map_location=device, 
#             weights_only=False
#         )

def load_solv_encoder(hw_setting):
    with torch.no_grad():
        heiwa = HEIWA3H(hw_setting=hw_setting)
    
        # heiwa = torch.load(
        #     path, 
        #     map_location="cuda", 
        #     weights_only=False
        # )
        # if hw_setting.device=="cuda":
        #     heiwa.to("cuda")
        orthogonalizer = HEIWA_Orthogonalizer(hw_setting.load_device)
        return heiwa, orthogonalizer

def load_thermo(hw_setting):
    thermo = ThermoPred(hw_setting=hw_setting)
    return thermo
