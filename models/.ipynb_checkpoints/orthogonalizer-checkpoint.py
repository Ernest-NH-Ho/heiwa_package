import torch.nn as nn
import torch
from heiwa.utils import asset_path

class HEIWA_Orthogonalizer:
    def __init__(self, device):
        self.avg = torch.load(asset_path.joinpath("avg1536_wt.pt"), map_location=device, weights_only=True)
        self.std = torch.load(asset_path.joinpath("sd1536_wt.pt"), map_location=device, weights_only=True)
        self.eigvec = torch.load(asset_path.joinpath("eigvec.pt"), map_location=device, weights_only=True)

    def to(self, device):
        self.avg = self.avg.to(device)
        self.std = self.std.to(device)
        self.eigvec = self.eigvec.to(device)
        
    def forward(self, x):
        x_out = (x-self.avg)/(self.std*1000)
        x_out = x_out @ (self.eigvec*2)
        return x_out

    def reverse(self, x):
        x_out = x @ torch.linalg.pinv(self.eigvec*2)
        x_out = (x_out * self.std*1000) + self.avg
        return x_out
