import torch
import numpy as np
import pickle
import traceback
from heiwa.readers.read_utils import read_coord, read_coord_coarse, extractNumbers
from heiwa.utils import AA
from esm.sdk.api import ESMProtein

idx = {}
# idx["ATOM"] = (0, 6)
idx["Chain"] = (21, 22)
idx["Resi"] = (22, 26)
idx["Resn"] = (17, 20)
idx["Atom_name"] = (11, 16)
idx["X-CA"] = (30, 38)
idx["Y-CA"] = (38, 46)
idx["Z-CA"] = (46, 54)


def check_pdb_format(row, idx):
    assert row[idx["Chain"][0]:idx["Chain"][1]] != " "
    assert row[idx["Resi"][0]:idx["Resi"][1]].isdigit()
    assert row[idx["Resn"][0]:idx["Resn"][1]] in AA
    for d in "XYZ":
        assert float(row[idx[f"{d}-CA"][0]:idx[f"{d}-CA"][1]])

def readPDB(
    filename: str, 
    cleave: int=0, 
    debug_path: str="", 
):
    """ 
    length is the length after STP truncation
    return seq, ss8, ACC, coords 
    retain_residues is assumes STP has been cut already """
    try:
        max_len=9999
        outputs = {}
        outputs['coordinates'] = np.full((max_len, 3, 3), np.nan)
        outputs["sequence"] = ["_" for _ in range(max_len)]
        chain = set()
        start_idx = max_len
        end_idx = 0
        
        
        with open(f"{filename.replace('.pdb', '')}.pdb", "r") as content:
            allrows = content.readlines()
            status = "wait"
            # print(f"Opened {filename}")
            
            for row in allrows:
                
                if status=="wait" and \
                (row.startswith("ATOM") or row.startswith("HETATM")):
                    status="read"
                    # print("Changed status to read")
    
                if status=="read" and row.startswith("ATOM"):
                    if debug_path:
                        check_pdb_format(row, idx)
                    # print(row)
                    if row[idx["Resi"][0]:idx["Resi"][1]].replace(" ", "").isdigit():
                        resi = int(row[idx["Resi"][0]:idx["Resi"][1]]) - cleave
                        # print(resi)
                        if resi > 0:
                            
                            # Resi
                            resi = int(extractNumbers(row, idx, "Resi"))
                            start_idx = min(start_idx, resi-1)
                            end_idx = max(end_idx, resi)
    
                            if resi - cleave > 0:
                                # chain
                                chain.add(extractNumbers(row, idx, "Chain"))
    
                                # is that CA
                                atom_name = extractNumbers(row, idx, "Atom_name").strip()
    
                                if atom_name == "CA":
                                    # AA
                                    aa = extractNumbers(row, idx, "Resn")
                                    outputs["sequence"][resi-1] = AA[aa]
                
                                    # Coords
                                    # print(row)
                                    # print(resi)
                                    outputs['coordinates'] = read_coord_coarse(row, outputs['coordinates'], 
                                                                               resi, 
                                                                               idx["X-CA"][0], idx["X-CA"][1], 
                                                                               idx["Y-CA"][0], idx["Y-CA"][1], 
                                                                               idx["Z-CA"][0], idx["Z-CA"][1], )
        # print(start_idx, end_idx, outputs["sequence"])
        outputs["sequence"] = "".join(outputs["sequence"][start_idx:end_idx])
        protein = ESMProtein(
            coordinates=torch.from_numpy(outputs['coordinates'][:len(outputs["sequence"]), ...]), 
            sequence=outputs["sequence"]
        )
        # print("a", type(protein))
        assert len(chain) <= 1
        assert outputs["sequence"].count("_")==0
        
        if debug_path:
                
            with open(debug_path, 'wb') as file:
                filename_base=filename[filename.rfind('/')+1:filename.rfind('.')]
                pickle.dump(protein, f"{debug_path}/pdb_output_from_{filename_base}.pkl")
            
        return protein
        
    except Exception as e:
        if filename.count('/') > 0:
            path = f"{filename[:filename.rfind('/')]}/error_when_read_pdb.txt"
        else:
            path = f"./error_when_read_pdb.txt"

        with open(path, "a+") as out:
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n\n") 