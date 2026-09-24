import heiwa
import torch
import numpy as np

from esm.models.esm3 import ESM3
with torch.no_grad():
    esm3_model = ESM3.from_pretrained(
        model_name="esm3_sm_open_v1", 
        device=torch.device("cuda")
    )

hw_setting = heiwa.Hardware_Setting(
    cpu=16, 
    batch_size=16, 
    gpu_rs=12345, 
    device="cpu"
)

with torch.no_grad():
    heiwa.gen_embed(
        esm3_model=esm3_model, 
        hw_setting=hw_setting, 
        inFile="../esm3_generation_with_heiwa3p/esm3_proteins/7repr_202605_192_224_heiwa1_softSASA_heiwaSeed4/AF-Q7SIG3-F1-model_v4_heiwa0_full.pdb", 
        outPath="gen_from_pdb/AF-Q7SIG3-F1-model_v4_heiwa0_full.pt", 
    )

with torch.no_grad():
    heiwa.gen_embed(
        esm3_model=esm3_model, 
        hw_setting=hw_setting, 
        inFile="test_101M/101M.cif", 
        outPath="gen_from_cif/101M.pt", 
    )

with torch.no_grad():
    heiwa.gen_embed(
        esm3_model=esm3_model, 
        hw_setting=hw_setting, 
        inFile="test_A0A0/AF-A0A009IHW8-F1-model_v4.dssp", 
        outPath="gen_from_dssp/AF-A0A009IHW8-F1-model_v4.pt", 
    )

with torch.no_grad():
    heiwa_model, heiwa_orthogonalizer = heiwa.load_solv_encoder(hw_setting=hw_setting)

# Q7SIG3
seq_prompt = "_"*43 + "H" + "_"*14 + "E" + "_" + "N" + "_"*2 + "T" + "_" + "E" + "_"*2 + "E" + "_"*23 + "D" + "_"*93 + "S" + "_"*49
print(len(seq_prompt))

with torch.no_grad():
    # 45, 59, 61, 64, 66, 69, 93, 187
    heiwa.de_novo(
        hw_setting=hw_setting, 
        esm3_model=esm3_model,
        heiwa_model=heiwa_model, 
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        cleave=0, 
        # seq_prompt=seq_prompt, 
        crit_resi = (45, 59, 61, 64, 66, 69, 93, 187), 
        idx_style=1, 
        inFile="../esm3_generation_with_heiwa3p/esm3_proteins/7repr_202605_192_224_heiwa1_softSASA_heiwaSeed4/AF-Q7SIG3-F1-model_v4_heiwa0_full.pdb", 
        outPath="denovo_from_pdb/AF-Q7SIG3-F1-model_v4_dn0.pdb", 
    )

# This protein has signal peptide
with torch.no_grad():
    heiwa.de_novo(
        hw_setting=hw_setting, 
        esm3_model=esm3_model,
        heiwa_model=heiwa_model, 
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        cleave=19, 
        # seq_prompt=seq_prompt, 
        crit_resi = (135, 143), 
        idx_style=1, 
        inFile="test_P0AA99/AF-P0AA99-F1-model_v6.pdb", 
        outPath="denovo_from_pdb_cleave/AF-P0AA99-F1-model_v6_dn19.pdb", 
    )

with torch.no_grad():
    thermo_model = heiwa.load_thermo(hw_setting=hw_setting)

# Load 1 tensor --> thermo
with torch.no_grad():
    tensor = torch.load(f"test_101M/101M.pt", map_location="cpu", weights_only=True)[1:-1, ...]
    heiwa.predict(
        # target="thermo", # "rmsf" or thermo
        heiwa_model=heiwa_model, 
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        thermo_model=thermo_model, 
        x=tensor, 
        # in_path, 
        # cleave, 
        out_folder="pred_from_1_tensor", 
        hw_setting=hw_setting, 
        exclude_cls=None, 
        exclude_eos=None, 
    )

# Load many path --> thermo
with torch.no_grad():
    # tensor = torch.load(f"101M.pt", map_location="cpu", weights_only=True)[1:-1, ...]
    heiwa.predict(
        # target="thermo", # "rmsf" or thermo
        heiwa_model=heiwa_model, 
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        thermo_model=thermo_model, 
        # x=tensor, 
        in_path=[
            "test_P0AA99/AF-P0AA99-F1-model_v6_dn19.pt", 
            "test_101M/101M.pt", 
            "test_A0A0/AF-A0A009IHW8-F1-model_v4.pt", 
            "test_Q7SIG3/AF-Q7SIG3-F1-model_v4_heiwa0_full.pt", 
        ], 
        cleave=[0, 0, 0, 0], 
        out_folder="pred_from_multi_path", 
        hw_setting=hw_setting, 
        exclude_cls=[True, True, True, True], 
        exclude_eos=[True, True, True, True], 
    )


# Load 1 path --> rmsf
with torch.no_grad():
    # tensor = torch.load(f"test_101M/101M.pt", map_location="cpu", weights_only=True)[1:-1, ...]
    heiwa.predict(
        # target="rmsf", # "rmsf" or thermo
        heiwa_model=heiwa_model, 
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        thermo_model=thermo_model, 
        # x=tensor, 
        in_path=f"test_101M/101M.pt", 
        cleave=0, 
        out_folder="pred_from_1_path", 
        hw_setting=hw_setting, 
        exclude_cls=True, 
        exclude_eos=True, 
    )

# Load many tensor --> rmsf
with torch.no_grad():
    x = []
    x.append(torch.load(f"test_P0AA99/AF-P0AA99-F1-model_v6_dn19.pt", map_location="cpu", weights_only=True)[1:-1, ...])
    x.append(torch.load(f"test_101M/101M.pt", map_location="cpu", weights_only=True)[1:-1, ...])
    x.append(torch.load(f"test_A0A0/AF-A0A009IHW8-F1-model_v4.pt", map_location="cpu", weights_only=True)[1:-1, ...])
    x.append(torch.load(f"test_Q7SIG3/AF-Q7SIG3-F1-model_v4_heiwa0_full.pt", map_location="cpu", weights_only=True)[1:-1, ...])
    heiwa.predict(
        # target="rmsf", # "rmsf" or thermo
        heiwa_model=heiwa_model, 
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        thermo_model=thermo_model, 
        x=x, 
        # in_path=[
        #     "test_P0AA99/AF-P0AA99-F1-model_v6_dn19.pt", 
        #     "test_101M/101M.pt", 
        #     "test_A0A0/AF-A0A009IHW8-F1-model_v4.pt", 
        #     "test_Q7SIG3/AF-Q7SIG3-F1-model_v4_heiwa0_full.pt", 
        # ], 
        cleave=None, 
        out_folder="pred_from_multi_tensor", 
        hw_setting=hw_setting, 
        exclude_cls=None, 
        exclude_eos=None, 
    )