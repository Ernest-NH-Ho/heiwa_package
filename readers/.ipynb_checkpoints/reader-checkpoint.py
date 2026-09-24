from heiwa.readers.read_cif import readCIF
from heiwa.readers.read_dssp import readDSSP
from heiwa.readers.read_pdb import readPDB

def read_file(
    filename:str, 
    cleave: int=0, 
    debug_path: str="", 
):
    if filename.endswith(".cif"):
        # print("Reading cif")
        return readCIF(
            filename=filename, 
            cleave=cleave, 
            debug_path=debug_path,
        )
    elif filename.endswith(".dssp"):
        print("Reading dssp")
        return readDSSP(
            filename=filename, 
            cleave=cleave, 
            debug_path=debug_path,
        )
    elif filename.endswith(".pdb"):
        # print(f"reading {filename}")
        return readPDB(
            filename=filename, 
            cleave=cleave, 
            debug_path=debug_path,
        )

    raise ValueError(f"{filename} --> File name must end with .cif .dssp .pdb")