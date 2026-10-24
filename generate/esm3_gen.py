import os
from copy import deepcopy
from heiwa._validation import single_source, nonnegative_int, prompt_options
import torch
import torch.nn as  nn
import pickle
import pandas as pd
# import traceback
from heiwa.readers.reader import read_file
from heiwa.generate.generate_utils import convert_idx_to_prompt
from esm.sdk.api import GenerationConfig, ESMProtein
# from heiwa.generate.modeller_script import fill_atom_to_pdb_in_single

tracks = [
    {'structure':16},
    {'secondary_structure':8}, 
    {'sasa':4}, 
    {'structure':8}, 
    {'sequence':8}
]

def esm3_generation(
    real_tracks, protein, 
    esm3_model, heiwa_model, heiwa_orthogonalizer, 
    seq_prompt, crit_resi, 
    out_filename
):
    if protein == "max_len_exceeded":
        pass
    fname = None
    for i, track in enumerate(real_tracks):
        # print(track)
        if i == len(real_tracks)-1:
            fname = out_filename
            
        track_k = list(track.keys())[0]
        track_v = list(track.values())[0]
        # print(track)
        # print(fname)
        condition_on_coordinates_only = True if track_k == 'structure' else False
        
        if track_k=='sequence':
            protein.sequence = seq_prompt
            # print(seq_prompt)
        elif track_k=='structure' and track_v == 8:
            protein = esm3_model.generate(protein, GenerationConfig(track=track_k, num_steps=track_v, \
                                                               temperature=0.7, 
                                                               temperature_annealing=True, \
                                                               condition_on_coordinates_only = condition_on_coordinates_only), \
                                     filename=fname, 
                                     layer_i=-1, 
                                     heiwa_model=heiwa_model, 
                                     heiwa_normalizer=heiwa_orthogonalizer)
            continue
        # print(len(protein.sequence), len(protein.coordinates), protein.sequence)
        protein = esm3_model.generate(protein, GenerationConfig(track=track_k, num_steps=track_v, \
                                                           temperature=0.7, 
                                                           temperature_annealing=True, \
                                                           condition_on_coordinates_only = condition_on_coordinates_only), \
                                 filename=fname, 
                                 layer_i=-1)
    return protein
    
def check_generation_success(outPath:str|list[str]):
    """Check where pdb and pkl exists at the right path"""
    if isinstance(outPath, str):
        with open(f"./generation_report.txt", "a+") as file:
            if os.path.exists(outPath):
                file.write(f"Generation of {outPath} successful.\n")
            else:
                file.write(f"Generation of {outPath} failed.\n")
    elif isinstance(outPath, list):
        with open(f"./generation_report.txt", "a+") as file:
            for path in outPath:
                if os.path.exists(path):
                    file.write(f"Generation of {path} successful.\n")
                else:
                    file.write(f"Generation of {path} failed.\n")

def save_protein(protein, folder_out, out_file_name):
    with open(f"{folder_out}/{out_file_name}.pkl", 'wb') as file:
        pickle.dump(protein, file)
    protein.to_pdb(f"{folder_out}/{out_file_name}.pdb")

def generate_proteins(
    task, 
    hw_setting, 
    esm3_model, 
    heiwa_model=None, heiwa_orthogonalizer=None, seq_prompt=None, crit_resi=None, idx_style:int|None=None, 
    inFile:str|None=None, 
    protein:ESMProtein|None=None, 
    cleave:int=0, 
    out_file_name:str|None=None, 
    folder_out=".", \
    debug_path: str="", 
):
    assert task in ("gen_embed", "gen_de_novo")
    assert esm3_model
    assert out_file_name
    single_source(inFile, protein)
    nonnegative_int(cleave, "cleave")
    assert (heiwa_model is None) == (heiwa_orthogonalizer is None), \
        "heiwa_model and heiwa_orthogonalizer must both be supplied or both be None"
        
    if task == "gen_embed":
        out_file_type="pt"
        real_tracks = tracks[:-1]
    else:
        assert seq_prompt is not None or (crit_resi is not None and idx_style is not None)
        out_file_type="pdb"
        real_tracks = tracks
        

    if inFile is not None:
        protein = read_file(
            filename=inFile, 
            cleave=cleave, 
            debug_path=debug_path, 
        )
    else:
        protein = deepcopy(protein)
        if cleave:
            for field in ("sequence", "coordinates", "secondary_structure", "sasa", "plddt"):
                value = getattr(protein, field, None)
                if value is not None:
                    setattr(protein, field, value[cleave:])
            if getattr(protein, "function_annotations", None):
                raise ValueError("Cleavage of protein function annotations is not supported")
    if protein is None or not protein.sequence:
        raise ValueError("Input must contain a nonempty protein sequence after cleavage")

    if task=="gen_de_novo" and seq_prompt is None:
        seq_prompt = convert_idx_to_prompt(
            idx=crit_resi, 
            cleave=cleave, 
            seq_wt=protein.sequence, 
            length=len(protein.sequence), 
            idx_style=idx_style, 
        )
        # print(seq_prompt)

    if task == "gen_de_novo":
        if len(seq_prompt) != len(protein.sequence):
            raise ValueError("seq_prompt length must match the protein after cleavage")
    hw_setting.switch_on_gpu_rs()
    
    if out_file_name.endswith(('.pt', '.pdb', '.pkl')):
        out_file_name = out_file_name.rsplit('.', 1)[0]
    if not os.path.exists(f"{folder_out}/{out_file_name}.{out_file_type}"):
        protein = esm3_generation(
            real_tracks=real_tracks, 
            protein=protein, 
            seq_prompt=seq_prompt, 
            crit_resi=crit_resi, 
            esm3_model=esm3_model, heiwa_model=heiwa_model, heiwa_orthogonalizer=heiwa_orthogonalizer, 
            out_filename=f"{folder_out}/{out_file_name}"
        )
        
        if task=="gen_de_novo":
            save_protein(protein=protein, 
                         folder_out=folder_out, 
                         out_file_name=out_file_name)
                
            # fill_atom_to_pdb_in_single(
            #     base_name=file_name, 
            #     folder_out=folder_out, 
            #     sequence=protein.sequence, 
            #     length=len(protein.sequence)
            # )
        

        

    
