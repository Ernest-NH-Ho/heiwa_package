import torch
import pandas as pd
import numpy as np
# from heiwa.data_containers import EpochData

class BatchDataAccumulator:
    """Class to collect data for a single epoch across all batches."""
    
    def __init__(self):
        # self.task = task
        self.data = {}
        # if epoch_config.loc["save_y_pred", self.task]:
        self.data['y_pred'] = torch.tensor([], dtype=torch.float32, device='cpu')
        # if epoch_config.loc["save_intermediate", self.task]:
        self.data['intermediate_output'] = torch.tensor([], dtype=torch.float32, device='cpu')
        self.data['mask'] = torch.tensor([], dtype=torch.float32, device='cpu') # Not for saving. Only for conversion of logits to rmsf values
        
    def add_batch(self, data_2_accum:dict):
        """ loss = is already weighted by weight_0 
        tp tn fp fn is not weighted by weight_0 """
        # if epoch_config.loc['save_y_pred', self.task]:
        self.data['y_pred'] = torch.cat((self.data['y_pred'], data_2_accum['y_pred'].cpu()), dim=0)

        # if epoch_config.loc['save_intermediate', self.task]:
        # for k in data_2_accum["intermediate_output"]:
        #     if not k in self.data['intermediate_output']:
        #         self.data['intermediate_output'][k] = torch.tensor([], dtype=torch.float32, device='cpu')
        self.data['intermediate_output'] = torch.cat((self.data['intermediate_output'], data_2_accum['intermediate_output'].cpu()), dim=0)
        if "mask" in data_2_accum:
            self.data['mask'] = torch.cat((self.data['mask'], data_2_accum['mask'].cpu()), dim=0)

    def save_data(self, out_folder):
        """ snapshots do not require accumulation """
        # if epoch_config.loc['save_y_pred', self.task]:
        y_pred = self.data['y_pred'].cpu().numpy()
        np.save(f"{out_folder}/thermophilicity.npy", y_pred)
            
        # if epoch_config.loc['save_intermediate', self.task]:
            # for k in self.data["intermediate_output"]:
            #     print(k, self.data["intermediate_output"][k].max())
            #     torch.save(self.data["intermediate_output"][k], f"{out_folder}/{k}.pt")
        self.data["intermediate_output"] = self.data["intermediate_output"].unsqueeze(-1) # this is for logit-rmsf conversion only
        self.data["intermediate_output"] = self.data["intermediate_output"].cpu().numpy()
        from heiwa.model_run.rmsf_convert import convert_logit_to_rmsf
        rsmf = convert_logit_to_rmsf(
            logits=self.data["intermediate_output"], 
        )
        
        if len(self.data['mask'].size())>1:
            self.data['mask'] = self.data['mask'].cpu().numpy()
            rsmf *= self.data['mask']  # mask out the padded positions into rmsf=0
        else:
            del self.data['mask']
        
        np.save(f"{out_folder}/residue_role_in_thermophilicity.npy", np.squeeze(self.data["intermediate_output"], axis=None))
        np.save(f"{out_folder}/rsmf.npy", rsmf)

        