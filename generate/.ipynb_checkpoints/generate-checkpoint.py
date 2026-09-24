import torch.nn as nn
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
    if outPath.count('/') > 0:
        sep = outPath.rfind('/')
        folder_out = outPath[:sep]
        outPath  = outPath[sep+1:]
    else:
        folder_out = "."
    ensure_dirs(folder_out)

    if inFile:
        generate_proteins(
            task="gen_embed", 
            hw_setting=hw_setting, 
            inFile=inFile, 
            out_file_name=outPath, 
            cleave=cleave, 
            esm3_model=esm3_model, 
            folder_out=folder_out, 
            debug_path=debug_path, 
        )
    else:
        assert protein
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
    assert not (seq_prompt is not None and (crit_resi is not None or idx_style))
    
    if outPath.count('/') > 0:
        sep = outPath.rfind('/')
        folder_out = outPath[:sep]
        outPath  = outPath[sep+1:]
    else:
        folder_out = "."
    ensure_dirs(folder_out)

    if heiwa_model:
        heiwa_model.move_to(hw_setting.run_device)
    if heiwa_orthogonalizer:
        heiwa_orthogonalizer.move_to(hw_setting.run_device)

    if inFile:
        generate_proteins(
            task="gen_de_novo", 
            hw_setting=hw_setting, 
            inFile=inFile, 
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
        assert protein
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