
def convert_idx_to_prompt(
    idx:list[int]|tuple[int], # only this is affected by signal peptide
    seq_wt:str, 
    length:int, 
    cleave:int=0, 
    idx_style=0, 
):
    seq_prompt = ""
    i_last=0
    # print(seq_wt)
    idx = sorted([i-cleave for i in idx])
    for i in idx:
        # print(i-i_last-idx_style, seq_wt[i-idx_style])
        seq_prompt += "_"*(i-i_last-idx_style) + seq_wt[i-idx_style]
        i_last = i
    seq_prompt += "_"*(length - i_last + 1 - idx_style)
    return seq_prompt