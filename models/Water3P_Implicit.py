import torch
import torch.nn as nn

class Water3P_Implicit(nn.Module):
    """ Use 3-point model, assume homogeneous at the external env (Poisson-Boltzmann's system) """
    """ It has its own parameters under this class """
    def __init__(self, dim_water:int, dim_qk:int, dim_v:int, num_head:int):
        super().__init__() 
        self.raw_emb = nn.Parameter(torch.randn(dim_water), requires_grad=True)
        self.raw_emb.pretrained = True
        self.dim_water = dim_water
        self.dim_v = dim_v
        self.num_head = num_head

        # K core
        self.K_H = nn.Linear(dim_water, dim_qk)
        self.K_L = nn.Linear(dim_water, dim_qk)
        self.K_L2 = nn.Linear(dim_water, dim_qk)
        self.K_O = nn.Linear(dim_water, dim_qk)
        
        # K angle
        self.Ka_H = nn.Linear(dim_water, 3)
        self.Ka_L = nn.Linear(dim_water, 3)
        self.Ka_O = nn.Linear(dim_water, 3)
        self.Ka_L2 = nn.Linear(dim_water, 3)
        self.angle_bias = nn.Parameter(torch.randn(num_head) * 0.125, requires_grad=True)

        # V
        self.V_H = nn.Linear(dim_water, dim_v)
        self.V_L = nn.Linear(dim_water, dim_v)
        self.V_O = nn.Linear(dim_water, dim_v)
        self.V_L2 = nn.Linear(dim_water, dim_v)

        
    def generate_k(self):
        kh = self.K_H(self.raw_emb)
        kl = self.K_L(self.raw_emb)
        ko = self.K_O(self.raw_emb)
        kl2 = self.K_L2(self.raw_emb)
        return torch.vstack([kh, kl, ko, kl2])

    def generate_ka(self):
        kh = self.Ka_H(self.raw_emb) # self.dropout_Ka_H()
        kl = self.Ka_L(self.raw_emb) # self.dropout_Ka_L()
        ko = self.Ka_O(self.raw_emb) # self.dropout_Ka_O()
        kl2 = self.Ka_L2(self.raw_emb) # self.dropout_Ka_O()
        return torch.vstack([kh, kl, ko, kl2])
        
    def generate_v(self):
        vh = self.V_H(self.raw_emb)
        vl = self.V_L(self.raw_emb)
        vo = self.V_O(self.raw_emb)
        vl2 = self.V_L2(self.raw_emb)
        return torch.vstack([vh, vl, vo, vl2])
