import torch
from heiwa.loaders.tensor_loader import concat_tensors_and_pad
from heiwa.loaders.dataloader import compile_data_single
from heiwa.hardware import Hardware_Setting
from heiwa.model_run.epoch_run import EpochRunner
from heiwa.utils import ensure_3d_tensor

def batch_run(
    x:torch.Tensor|list[torch.Tensor], 
    hw_setting:Hardware_Setting|None=None, 
    heiwa_model=None, 
    heiwa_orthogonalizer=None, 
    thermo_model=None, 
    out_folder=".", 
):
    hw_setting = hw_setting if hw_setting is not None else Hardware_Setting()
    # assert target.lower() in ("rmsf", "thermo")
    
    # ========= Prepare dataset here =========
    # m = None
    # l = None
    if isinstance(x, list):
        dataset = concat_tensors_and_pad(
            embeds=x, 
            # create_l=target.lower()=="thermo", 
            hw_setting=hw_setting, 
        )
        # print("x_unorthogonalized_list", len(x), x)
        # print("x_unorthogonalized", dataset["X"].size(), dataset["X"])
        # print("l", l.size(), l)
    else:
        x = ensure_3d_tensor(x=x)
        # if target.lower()=="thermo":
        #     print(f"Task is definitely thermo")
        l = torch.tensor(
            [x.size(1)] * x.size(0), 
            dtype=torch.float32, 
            device='cpu'
        ) 
        dataset={"X":x, "L":l}
    print("dataset", dataset.keys())

    # ========= Prepare models, hardware, epoch_runner =========
    assert heiwa_model is not None and thermo_model is not None

    hw_setting.switch_on_gpu_rs()

    epoch_runner = EpochRunner(out_folder=out_folder)
    

    data_loader = compile_data_single(
        hw_setting=hw_setting, 
        dataset=dataset, 
    )
        
    # ========= Run ==========================
    print(f"Start running batches")
    
    # print("dataset", data_loader.dataset)
    epoch_runner.run(
        data_source=data_loader,
        heiwa_model=heiwa_model,
        heiwa_orthogonalizer=heiwa_orthogonalizer, 
        thermo_model=thermo_model, 
        hw_setting=hw_setting
    )
