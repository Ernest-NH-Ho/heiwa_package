import torch.nn as nn
from pathlib import Path
from heiwa._validation import single_source, nonnegative_int, prompt_options
from esm.sdk.api import ESMProtein
from heiwa.generate.esm3_gen import generate_proteins
from heiwa.utils import ensure_dirs

def gen_embed(
    esm3_model, 
    hw_setting, 
    inFile:str|None=None, 
    protein:ESMProtein|None=None, 
    cleave:int=0, 
    outPath:str="./no_name.pt", 
    debug_path:str="", 
):
    """1 protein at once"""
    single_source(inFile, protein)
    nonnegative_int(cleave, "cleave")
    output = Path(outPath)
    folder_out, outPath = str(output.parent), output.name
    ensure_dirs(folder_out)

    if inFile is not None:
        generate_proteins(
            task="gen_embed", 
            hw_setting=hw_setting, 
            inFile=str(inFile), 
            out_file_name=outPath, 
            cleave=cleave, 
            esm3_model=esm3_model, 
            folder_out=folder_out, 
            debug_path=debug_path, 
        )
    else:
        generate_proteins(
            task="gen_embed", 
            hw_setting=hw_setting, 
            protein=protein, 
            cleave=cleave, 
            out_file_name=outPath, 
            esm3_model=esm3_model, 
            folder_out=folder_out, 
            debug_path=debug_path, 
        )

def de_novo(
    esm3_model, 
    hw_setting, 
    heiwa_model:nn.Module|None=None, 
    heiwa_orthogonalizer:nn.Module|None=None, 
    inFile:str|None=None, 
    seq_prompt:str|None=None, 
    crit_resi:list[int]|None=None, 
    idx_style:int|None=None, 
    cleave:int=0, 
    protein=None, 
    outPath:str="./no_name.pdb", 
    debug_path:str="", 
):
    """1 protein at once"""
    prompt_options(seq_prompt, crit_resi, idx_style)
    assert (heiwa_model is None) == (heiwa_orthogonalizer is None), \
        "heiwa_model and heiwa_orthogonalizer must both be supplied or both be None"
    
    single_source(inFile, protein)
    nonnegative_int(cleave, "cleave")
    output = Path(outPath)
    folder_out, outPath = str(output.parent), output.name
    ensure_dirs(folder_out)

    if heiwa_model:
        heiwa_model.move_to(hw_setting.run_device)
        assert heiwa_orthogonalizer is not None
        heiwa_orthogonalizer.move_to(hw_setting.run_device)

    if inFile is not None:
        generate_proteins(
            task="gen_de_novo", 
            hw_setting=hw_setting, 
            inFile=str(inFile), 
            cleave=cleave, 
            seq_prompt=seq_prompt, 
            crit_resi=crit_resi, 
            idx_style=idx_style, 
            out_file_name=outPath, 
            esm3_model=esm3_model, 
            heiwa_model=heiwa_model, 
            heiwa_orthogonalizer=heiwa_orthogonalizer, 
            folder_out=folder_out, 
            debug_path=debug_path, 
        )
    else:
        generate_proteins(
            task="gen_de_novo", 
            hw_setting=hw_setting, 
            protein=protein, 
            cleave=cleave, 
            seq_prompt=seq_prompt, 
            crit_resi=crit_resi, 
            idx_style=idx_style, 
            out_file_name=outPath, 
            esm3_model=esm3_model, 
            heiwa_model=heiwa_model, 
            heiwa_orthogonalizer=heiwa_orthogonalizer, 
            folder_out=folder_out, 
            debug_path=debug_path, 
        )
