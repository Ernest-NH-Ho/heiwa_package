import torch
import numpy as np
import pickle
import traceback
from CifFile import ReadCif
from heiwa.utils import AA
from esm.sdk.api import ESMProtein
                

def check_pdb_format(row, idx):
    assert row[idx["Chain"][0]:idx["Chain"][1]] != " "
    assert row[idx["Resi"][0]:idx["Resi"][1]].isdigit()
    assert row[idx["Resn"][0]:idx["Resn"][1]] in AA
    for d in "XYZ":
        assert float(row[idx[f"{d}-CA"][0]:idx[f"{d}-CA"][1]])

def readCIF(
    filename: str, 
    cleave: int=0, 
    debug_path: str="", 
):
    """ cif """
    try:
        cf = ReadCif(filename)
        assert len(cf.keys()) == 1
        block_name = list(cf.keys())[0]
        block = cf[block_name]
    
        chain = set()
        outputs = {}
        max_len=9999
        outputs['coordinates'] = np.full((max_len, 3, 3), np.nan)
        outputs["sequence"] = ["_" for _ in range(max_len)]
        # assert all([ for aa in block["_atom_site.label_comp_id"]])
        
        min_resi = min([int(i) for i in block["_atom_site.label_seq_id"] if i.isdigit()])
        end_idx = 0

        # print(f"Inside readCIF")
        for head, atom_type, aa, chain_id, resi, x, y, z in zip(
            block["_atom_site.group_PDB"], 
            block["_atom_site.label_atom_id"], 
            block["_atom_site.label_comp_id"], 
            block["_atom_site.label_asym_id"], 
            block["_atom_site.label_seq_id"], 
            block["_atom_site.Cartn_x"], 
            block["_atom_site.Cartn_y"], 
            block["_atom_site.Cartn_z"], 
        ):
            if head=="ATOM" and atom_type=="CA":
                # print(head, atom_type, aa, chain_id, resi, x, y, z)
                
                chain.add(chain_id)
                resi = int(resi)
                assert aa in AA
                if resi - min_resi - cleave >= 0:
                    outputs["sequence"][resi-min_resi] = AA[aa]
                    outputs['coordinates'][resi-min_resi, 0, :] = float(x), float(y), float(z)
                    outputs['coordinates'][resi-min_resi, 1, :] = float(x), float(y), float(z)
                    outputs['coordinates'][resi-min_resi, 2, :] = float(x), float(y), float(z)
                    end_idx = max(end_idx, resi - min_resi + 1)
                    
        outputs["sequence"] = "".join(outputs["sequence"][cleave:end_idx])
        protein = ESMProtein(
            coordinates=torch.from_numpy(outputs['coordinates'][cleave:end_idx, ...]), 
            sequence=outputs["sequence"]
        )

        assert len(chain) <= 1
        # print(protein.sequence)
        assert protein.sequence.count("_")==0
        
        if debug_path:
            with open(debug_path, 'wb') as file:
                filename_base=filename[filename.rfind('/')+1:filename.rfind('.')]
                pickle.dump(protein, file)
        
        return protein
        
    except Exception as e:
        if filename.count('/') > 0:
            path = f"{filename[:filename.rfind('/')]}/error_when_read_cif.txt"
        else:
            path = f"./error_when_read_cif.txt"

        with open(path, "a+") as out:
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n\n")
    
        raise
