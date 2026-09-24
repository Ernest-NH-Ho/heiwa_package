import os
import traceback
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from heiwa.hardware import Hardware_Setting
from tqdm import tqdm
from heiwa.utils import package_path
from heiwa.model_run.batch_accumulator import BatchDataAccumulator


class EpochRunner:
    def __init__(self, out_folder=None):
        self.out_folder = out_folder
        # Initialize BCE loss
        # self.config = pd.read_csv(
        #     package_path.joinpath("model_run", "epoch_config.csv"), 
        #     header=0, index_col='task'
        # ).astype(bool)
        torch.set_grad_enabled(False)
        print("Epoch runner initialized")
    
    def move_batch_2_device(self, batch, device):
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                batch_device[key] = {k: v.to(device) for k, v in value.items()}
            else:
                batch_device[key] = value.to(device)
        return batch_device
        
    def forward_pass(self, heiwa_model, heiwa_orthogonalizer, thermo_model, batch_device):
        # Forward pass
        with torch.amp.autocast('cuda'):
            batch_device['model_input']["X"] = heiwa_orthogonalizer.forward(batch_device['model_input']["X"])
            heiwa_output = heiwa_model(x=batch_device['model_input']["X"])
            model_output = thermo_model(
                # target=task, 
                model_input=batch_device['model_input'], 
                water_modification=heiwa_output, 
            )
            # print(model_output.y_pred)
            # if torch.isnan(model_output.y_pred).any() or torch.isinf(model_output.y_pred).any():
            #     print("NaN/Inf in y_pred")
            #     # quit()
            # if len(model_output.y_pred.size()) == 0:
            #     print(f"Failed to produce prediction")
            #     print(len(model_output.y_pred.size()))
            #     print(model_output.y_pred)
            #     # quit()
            # print(model_output.y_pred.size())
            return model_output

    def run(
        self, 
        data_source: DataLoader, 
        heiwa_model: nn.Module, 
        heiwa_orthogonalizer: nn.Module, 
        thermo_model: nn.Module, 
        hw_setting: Hardware_Setting|None

    ):
        """    For heiwa package      """

        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        if hw_setting:
            device = hw_setting.run_device
        else:
            device="cuda"
        
        # Move model to device
        heiwa_model.move_to(device)
        thermo_model.move_to(device)

        # Progress bar
        print(f"Creating tqdm pbar and batch_data accumulator")
        pbar = tqdm(data_source, desc=f"Batch run")

        data_accumulator = BatchDataAccumulator()
        #     task=task, 
        #     epoch_config=self.config, 
        # )
        
        for i, batch in enumerate(pbar):
            # Initialize data accumulator, if required
            data_2_accum = {}

            # Move batch to device
            batch_device = self.move_batch_2_device(batch=batch, device=device)
                
            # ================ Forward pass =======================
            model_output = self.forward_pass(
                heiwa_model=heiwa_model, 
                heiwa_orthogonalizer=heiwa_orthogonalizer, 
                thermo_model=thermo_model, 
                batch_device=batch_device
            )

            # Accumulate y_pred y_true
            # if self.config.loc["save_y_pred", task].any():
            data_2_accum['y_pred'] = model_output.y_pred
                    
            # Accumulate intermediate layer
            # if self.config.loc["save_intermediate", task].any():
            data_2_accum['intermediate_output'] = model_output.intermediate_output
            if "M" in batch_device["model_input"]:
                data_2_accum['mask'] = batch_device["model_input"]["M"]
                
            # Accumulate data
            # if self.config[task].any():
            data_accumulator.add_batch(
                data_2_accum=data_2_accum, 
                # epoch_config=self.config, 
            )

        # This part will convert to rmsf, if needed
        data_accumulator.save_data(
            # epoch_config=self.config, 
            out_folder=self.out_folder
        )
        
