import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict
from heiwa.hardware import Hardware_Setting

class MyDataset(Dataset):
    def __init__(self, prefix):
        """ The real initlization is called by cls._from_dict """
        self.DATAFILE_PREFIX = tuple(prefix)
        print("Tensor prefix", self.DATAFILE_PREFIX)
        
    @classmethod
    def _from_dict(cls, all_inputs:dict):
        """ This triggers __init__ """
        dataset = MyDataset(prefix=all_inputs.keys())
        for key, value in all_inputs.items():
            setattr(dataset, key, value)
            if key == "X":
                print(f"{key}: {len(value)}")
            else:
                print(f"{key}: {value}")
        return dataset

    def __str__(self):
        cout = ""
        for k in "XLM":
            cout += f"{k}:{getattr(self, k, None)}\n"
        return cout
        
    def __len__(self) -> int:
        if hasattr(self, "X"):
            return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        output = {}
        output['model_input'] = {k: getattr(self, k)[idx] for k in self.DATAFILE_PREFIX}
        return output



def compile_data_single(
    hw_setting:Hardware_Setting, 
    dataset:dict
) -> DataLoader:
    """     Create DataLoader objects for train, validation, and test sets.
    
    Args:
        target: 'test', 'inference', 'train' if fisher
        dataset: dictionary of individual raw tensors
        params: all Parameters
    
    Returns:
        output_dataset      """
    # Create datasets
    # print(dataset)
    output_dataset = MyDataset._from_dict(
        dataset # {k:dataset[k] for k in dataset}
    ) # 
        
    output_dataset = DataLoader(
        output_dataset,
        batch_size=hw_setting.batch_size,
        shuffle=False,
        num_workers=hw_setting.cpu,
        pin_memory=True
    )
    print(f"Dataloader for compiled")
    return output_dataset
    