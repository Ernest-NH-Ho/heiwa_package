# from dataclasses import dataclass
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ..hardware import Hardware_Setting

# @dataclass
# class Loaded_Tensor:
#     tensor: torch.tensor
#     length: int|None
    
def trim_tensor(tensor, exclude_cls, exclude_eos):
    with torch.no_grad():
        if isinstance(tensor, torch.Tensor):
            if exclude_cls:
                tensor = tensor[1:, ...]
            if exclude_eos:
                tensor = tensor[:-1, ...]
            return tensor
        else:
            assert len(tensor) == len(exclude_cls) == len(exclude_eos)
            # with ThreadPoolExecutor(max_workers=hw_setting.cpu) as executor:
            #     futures = [executor.submit(trim_tensor, tens, excl_cls, excl_eos) \
            #                ]
            #     for fut in futures:
            #         fut.result()
            for i, (tens, excl_cls, excl_eos) in enumerate(zip(tensor, exclude_cls, exclude_eos)):
                tensor[i] = trim_tensor(
                    tens, excl_cls, excl_eos
                )
            return tensor
        
    
def load_single_tensor(path:str, exclude_cls=False, exclude_eos=False): # , output_len=False
    with torch.no_grad():
        # construct_M:bool, 
        assert path.endswith(".npy") or path.endswith(".pt")
        with torch.no_grad():
            if path.endswith(".npy"):
                tensor = torch.from_numpy(
                    np.load(
                        path, 
                        allow_pickle=True
                    )
                )
            else:
                tensor = torch.load(
                    path, 
                    map_location="cpu", 
                    weights_only=True, 
                )
                
            tensor = trim_tensor(
                tensor=tensor, 
                exclude_cls=exclude_cls, 
                exclude_eos=exclude_eos
            )
            
            return tensor # Loaded_Tensor(tensor=tensor)


def concat_tensors_and_pad(
    embeds:list, 
    # create_l:bool, 
    hw_setting:Hardware_Setting
):
    with torch.no_grad():
        dataset= {}
        bs = len(embeds)
        # print(bs)
        l = torch.tensor(
            [embed.size(0) for embed in embeds], 
            dtype=torch.int32, device="cpu"
        )
        length_max = max(l)
        embed_concat = torch.zeros((bs, length_max, 1536)).float()
        
        def copy_to_new_tensor(i):
            embed_concat[i, :l[i], ...] = embeds[i][:l[i], ...]
    
        mask = torch.arange(length_max).expand(bs, length_max) < l.unsqueeze(1)
        
        with ThreadPoolExecutor(max_workers=hw_setting.cpu) as executor:
            futures = [executor.submit(copy_to_new_tensor, i) for i in range(len(embeds))]
            for fut in futures:
                fut.result()
        dataset["X"] = embed_concat
        dataset["M"] = mask
        dataset["L"] = l
        return dataset
            