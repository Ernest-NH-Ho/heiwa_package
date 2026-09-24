import torch
import numpy as np
import pickle
import traceback
from heiwa.readers.read_utils import read_coord, read_coord_coarse, extractNumbers
from esm.sdk.api import ESMProtein

def get_index(title:str):
    idx = {}
    idx["Resi"] = (title.find("RESIDUE"), title.find("RESIDUE")+5)
    idx["Resn"] = (title.find("AA"), title.find("AA")+1)
    idx["X-CA"] = (title.find("X-CA")-2, title.find("X-CA")+4)
    idx["Y-CA"] = (title.find("Y-CA")-2, title.find("Y-CA")+4)
    idx["Z-CA"] = (title.find("Z-CA")-2, title.find("Z-CA")+4)
    return idx

def readDSSP(
    filename: str, 
    cleave: int=0, 
    debug_path: str="", 
):
    """ dssp_name is assumed to be f"AF-{entry}-F1-model_v4.dssp"
    length is the length after STP truncation
    return seq, ss8, ACC, coords 
    retain_residues is assumes STP has been cut already """
    outputs = {}
    outputs['coordinates'] = np.full((9999, 3, 3), np.nan)
    outputs["sequence"] = ""
    
    length=0
    with open(f"{filename.replace('.dssp', '')}.dssp", "r") as content:
        allrows = content.readlines()
        status = "wait"
        
        for row in allrows:
            
            if row.count(" # ") > 0:
                idx = get_index(row)
                x_start, x_end = idx["X-CA"]
                y_start, y_end = idx["Y-CA"]
                z_start, z_end = idx["Z-CA"]
                status = "read"
                # print(row)
                
            elif status == "read":
                start, end = idx["Resi"]
                
                if row[start:end].replace(" ", "").isdigit():
                    resi = int(row[start:end]) - cleave
                    if resi > 0:
                        # AA
                        resn = extractNumbers(row, idx, "Resn")
                        outputs["sequence"] += resn
    
                        # Coords
                        # print(row)
                        # print(resi)
                        outputs['coordinates'] = read_coord_coarse(row, outputs['coordinates'], 
                                                                   resi, 
                                                                   x_start, x_end, 
                                                                   y_start, y_end, 
                                                                   z_start, z_end)
                        length+=1
                    
                        
    protein = ESMProtein(
        coordinates=torch.from_numpy(outputs['coordinates'][:length, ...]), 
        sequence=outputs["sequence"]
    )

    if debug_path:
        with open(debug_path, 'wb') as file:
            filename_base=filename[filename.rfind('/')+1:filename.rfind('.')]
            pickle.dump(protein, f"{debug_path}/dssp_output_from_{filename_base}.pkl")
        
    return protein