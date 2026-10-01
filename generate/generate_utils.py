from heiwa._validation import nonnegative_int, prompt_options


def convert_idx_to_prompt(idx, seq_wt, length, cleave=0, idx_style=0):
    """Indices refer to the original sequence; seq_wt is already cleaved."""
    prompt_options(None, idx, idx_style)
    nonnegative_int(cleave, "cleave")
    if length != len(seq_wt):
        raise ValueError("length must match seq_wt")
    prompt = ["_"] * length
    for index in idx:
        position = index - idx_style - cleave
        if not 0 <= position < length:
            raise ValueError(f"Critical residue {index} is outside the retained sequence")
        prompt[position] = seq_wt[position]
    return "".join(prompt)
