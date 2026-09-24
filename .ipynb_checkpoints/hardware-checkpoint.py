from dataclasses import dataclass
import torch
# import numpy as np

@dataclass
class Hardware_Setting:
    cpu: int = 1
    batch_size : int = 2
    gpu_rs: int|None = None
    load_device: str = "cpu"
    run_device: str = "cuda"

    def __str__(self):
        cout = "HEIWA hardware setting:\n"
        cout += f"cpu = {self.cpu}\n"
        cout += f"batch_size = {self.batch_size}\n"
        cout += f"gpu_rs = {self.gpu_rs}\n"
        cout += f"load_device = {self.load_device}\n"
        cout += f"run_device = {self.run_device}\n"
        # cout += f"cpu_rs = {self.cpu_rs}\n"
        return cout

    def switch_on_gpu_rs(self):
        if self.gpu_rs is not None:
            torch.manual_seed(self.gpu_rs)
        
    