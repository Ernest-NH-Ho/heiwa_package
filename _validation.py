from os import PathLike, fspath


def nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def single_source(inFile, protein):
    if (inFile is None) == (protein is None):
        raise ValueError("Supply exactly one of inFile or protein; the other must be None")
    if inFile is not None:
        if not isinstance(inFile, (str, PathLike)) or not fspath(inFile):
            raise ValueError("inFile must be a single nonempty file path")


def prompt_options(seq_prompt, crit_resi, idx_style):
    if seq_prompt is not None:
        if crit_resi is not None or idx_style is not None:
            raise ValueError("With seq_prompt, crit_resi and idx_style must be None")
        if not isinstance(seq_prompt, str) or not seq_prompt:
            raise ValueError("seq_prompt must be a nonempty string")
    else:
        if not isinstance(crit_resi, (list, tuple)) or idx_style is None:
            raise ValueError("Supply crit_resi and idx_style when seq_prompt is None")
        if type(idx_style) is not int or idx_style not in (0, 1):
            raise ValueError("idx_style must be 0 or 1")
        for index in crit_resi:
            nonnegative_int(index, "critical residue index")
