"""Prediction from either prepared tensors or embedding files."""
from os import PathLike, fspath
import torch
from heiwa.loaders.tensor_loader import load_single_tensor
from heiwa.batch_run import batch_run
from heiwa.hardware import Hardware_Setting
from heiwa.utils import ensure_dirs, record_input_lst


def _embedding(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Each embedding must be a torch.Tensor")
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != 1536 or any(n == 0 for n in tensor.shape):
        raise ValueError("Embeddings must have nonempty shape (L, 1536) or (B, L, 1536)")
    return tensor


def predict(heiwa_model, heiwa_orthogonalizer, thermo_model, x=None,
            in_path=None, out_folder=".", hw_setting=None,
            exclude_cls=None, exclude_eos=None):
    """Choose x OR in_path. File metadata must be None with x.

    All embeddings must already be cleaved. File mode requires both exclusion
    flags; for multiple files, supply equally sized lists of flags. Direct
    tensors must also have CLS/EOS tokens removed as needed, and both flags
    must be None.
    """
    if (x is None) == (in_path is None):
        raise ValueError("Supply exactly one of x or in_path; the other must be None")
    paths = None
    if x is not None:
        if any(value is not None for value in (exclude_cls, exclude_eos)):
            raise ValueError("With x, exclude_cls and exclude_eos must both be None")
        if isinstance(x, list):
            if not x:
                raise ValueError("x must not be empty")
            for tensor in x:
                _embedding(tensor)
                if tensor.ndim != 2:
                    raise ValueError("Each tensor in x must have shape (L, 1536)")
        else:
            _embedding(x)
    else:
        if isinstance(in_path, (str, PathLike)):
            paths = [fspath(in_path)]
            cls_flags, eos_flags = [exclude_cls], [exclude_eos]
        elif isinstance(in_path, (list, tuple)) and in_path:
            paths = list(in_path)
            for name, values in (("exclude_cls", exclude_cls), ("exclude_eos", exclude_eos)):
                if not isinstance(values, (list, tuple)) or len(values) != len(paths):
                    raise ValueError(f"{name} must have one entry per input path")
            cls_flags, eos_flags = exclude_cls, exclude_eos
        else:
            raise ValueError("in_path must be a file path or a nonempty ordered list of paths")
        for path, cls, eos in zip(paths, cls_flags, eos_flags):
            if not isinstance(path, (str, PathLike)) or not fspath(path):
                raise ValueError("Each input path must be a nonempty file path")
            if type(cls) is not bool or type(eos) is not bool:
                raise ValueError("exclude_cls and exclude_eos must be boolean")
        paths = [fspath(path) for path in paths]
        x = []
        for path, cls, eos in zip(paths, cls_flags, eos_flags):
            tensor = _embedding(load_single_tensor(path, exclude_cls=cls, exclude_eos=eos))
            if tensor.ndim != 2:
                raise ValueError("Each embedding file must contain shape (L, 1536)")
            x.append(tensor)
    if any(model is None for model in (heiwa_model, heiwa_orthogonalizer, thermo_model)):
        raise ValueError("All three prediction models are required")
    hw_setting = hw_setting if hw_setting is not None else Hardware_Setting()
    ensure_dirs(out_folder)
    batch_run(x=x, heiwa_model=heiwa_model, heiwa_orthogonalizer=heiwa_orthogonalizer,
              thermo_model=thermo_model, hw_setting=hw_setting, out_folder=out_folder)
    if paths is not None:
        record_input_lst(inFiles=paths, out_folder=out_folder)
