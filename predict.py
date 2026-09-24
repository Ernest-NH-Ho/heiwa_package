import torch
import torch.nn as nn
import numpy as np
import os
from heiwa.loaders.tensor_loader import load_single_tensor
from heiwa.batch_run import batch_run
from heiwa.hardware import Hardware_Setting
from heiwa.utils import ensure_dirs, record_input_lst #, isiterable

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def predict(
    heiwa_model, 
    heiwa_orthogonalizer, 
    thermo_model, 
    x:torch.Tensor|list|None=None, 
    in_path:str|list|None=None, 
    cleave:list[int]|None=None, 
    out_folder:str=".", 
    hw_setting: Hardware_Setting|None=None, 
    exclude_cls:bool|list=False, 
    exclude_eos:bool|list=False, 
):
    print(f"Prediction of 1. protein thermophilicity (Acc=0.88, F1=0.88, Prec=0.90, Sens=0.86, Spec=0.91)")
    print(f"2. Individual residue contribution to thermophilicity, and")
    print(f"3. Residue rmsf.")
    print()
    print(f"WARNING: the rmsf prediction by heiwa is meant to be rough estimates.")
    print(f"Temperature is not part of the model input. The author interpret the values in this way:")
    print(f"  - The absoluate value is not accurate")
    print(f"  - Only the relative values comparing different residues of the same sequence are more meaningful")
    print(f"  - All rmsf estimates assume intact protein structure. Not for simulating protein melt.")
    print()
    print(f"The following assumptions applies to heiwa:")
    print(f"  - pH ~= 7")
    print(f"  - Protein resides not in extreme salinity, pressure, or high wine concentration")
    print()
    assert x is not None or in_path is not None # one of them must be given
    # if x is not None:
    #     assert exclude_cls != True and exclude_eos != True # If tensor is given, I assume u trimmed it properly

    ensure_dirs(out_folder)
    
    if x is not None:
        if exclude_cls is not None or exclude_eos is not None:
            raise ValueError("If directly supply tensor(s), exclude_cls and exclude_eos must be None")
        if cleave is not None:
            raise ValueError("If directly supply tensor(s), cleave must be None")
        if in_path is not None:
            raise ValueError("If directly supply tensor(s), in_path must be None")
            
        
        batch_run(
            x=x, 
            heiwa_model=heiwa_model,
            heiwa_orthogonalizer=heiwa_orthogonalizer, 
            thermo_model=thermo_model, 
            hw_setting=hw_setting, 
            out_folder=out_folder, 
        )
        
    elif in_path is not None:
        if cleave is None:
            raise ValueError("If supplying path as input, cleave number must not be None")
        if exclude_cls is None or exclude_eos is None:
            raise ValueError("If supplying path as input, exclude_cls and exclude_eos must be boolean")
        if x is not None:
            raise ValueError("If supplying path as input, tensor input x must be None")
            
        if isinstance(in_path, str):
            x = load_single_tensor(
                path=in_path, 
                exclude_cls=exclude_cls, 
                exclude_eos=exclude_eos, 
            )
            batch_run(
                x=x, 
                heiwa_model=heiwa_model, 
                heiwa_orthogonalizer=heiwa_orthogonalizer, 
                thermo_model=thermo_model, 
                hw_setting=hw_setting, 
                out_folder=out_folder, 
            )
            
        else:
            assert isinstance(in_path, list|tuple|set)
            assert len(exclude_cls) == len(exclude_eos) == len(in_path)
            x = []
            for i, path in enumerate(in_path):
                
                tensor = load_single_tensor(
                                path=path, 
                                exclude_cls=exclude_cls[i], 
                                exclude_eos=exclude_eos[i], 
                            )
                x.append(tensor)
            # print(f"After import: len(x) = {len(x)}")
            # lengths.append(l)
        
            batch_run(
                x=x, 
                heiwa_model=heiwa_model, 
                heiwa_orthogonalizer=heiwa_orthogonalizer, 
                thermo_model=thermo_model, 
                hw_setting=hw_setting, 
                out_folder=out_folder, 
            )

            record_input_lst(
                inFiles=in_path, 
                out_folder=out_folder
            )
    
