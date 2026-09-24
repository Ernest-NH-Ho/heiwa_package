import torch
import torch.nn as nn
import torch.nn.functional as F
from heiwa.models.file_handling import compress, decompress
from heiwa.models.basic_layers import Swish
from heiwa.models.data_containers import ModelOutput
from heiwa.utils import asset_path

class ExposedResiduePrediction(nn.Module):
    """ Can be used for  """
    def __init__(self, dim_prot, dim_intermediate, dim_out):
        # dim_intermediate1, dim_intermediate2, 
        super().__init__()
        self.swish = Swish(in_dim=dim_prot, out_dim=dim_intermediate)
        self.ffn = nn.Linear(dim_intermediate, dim_out)

    def _load_default(self, num_cpu, device):
        path = asset_path.joinpath(f"ExposedResiduePrediction_epoch_final_model.pth")
        decompress(f"{path}", num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location=device, weights_only=True)
        incompatible = self.load_state_dict(state_dict)
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", num_cpu=num_cpu)
        
    def forward(self,  x, dataset_num=""):
        bs, l, _ = x.size()
        x = self.swish(x)
        x = self.ffn(x).reshape(bs, l) # * model_input['M_'+dataset_num]
        if self.training:
            x = torch.clamp(x, max=7.5, min=-7.5)
        return ModelOutput(y_pred=F.sigmoid(x))

