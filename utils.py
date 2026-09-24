from importlib.resources import files
import os
import torch
import numpy as np
import datetime

AA = {
    "ALA":"A", "CYS":"C", "ASP":"D", "GLU":"E", "PHE":"F", 
    "GLY":"G", "HIS":"H", "ILE":"I", "LYS":"K", "LEU":"L", 
    "MET":"M", "ASN":"N", "PRO":"P", "GLN":"Q", "ARG":"R", 
    "SER":"S", "THR":"T", "SEC":"U", "VAL":"V", "TRP":"W", "TYR":"Y"
}

package_path = files("heiwa")
asset_path = package_path.joinpath("model_assets")


def ensure_dirs(folder_path):
    # Create directories if they don't exist
    os.makedirs(folder_path, exist_ok=True)\

def ensure_3d_tensor(x):
    """ useful when running just 1 sample: 
    tensor.size() change from (seq_len, 1536) to (1, seq_len, 1536) """
    if len(x.size()) == 2:
        x = x.unsqueeze(0)
    elif len(x.size()) < 2:
        raise ValueError(f"Dimension of x tensor < 2") 
    
    # if m is not None:
    #     if len(m.size()) == 2:
    #         m = m.unsqueeze(0)
    #     elif len(m.size()) < 2:
    #         raise ValueError(f"Dimension of m tensor < 2") 

    # if len(l.size()) == 1:
    #     m = m.reshape(1, -1, 1)
    
    return x # , l # , m

def record_input_lst(inFiles:list, out_folder:str):
    now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    with open(f"{out_folder}/inputs_{now}.txt", "w") as file:
        file.write("\n".join(inFiles))

def isiterable(obj):
    try:
        iter(obj)
        if isinstance(obj, torch.Tensor|np.array|list|tuple|set):
            return False
        return True
    except TypeError:
        return False
