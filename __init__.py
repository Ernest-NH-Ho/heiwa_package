# Copyright (c) 2026 Ernest Ngai Hei Ho. Licensed under the MIT License.
__version__ = "0.1.2"


def _install_esm_backend():
    import importlib
    import importlib.util
    from pathlib import Path
    import sys

    root = Path(__file__).parent / "replace_esm3"
    for name, filename in (
        ("esm.layers.blocks", "blocks.py"),
        ("esm.layers.transformer_stack", "transformer_stack.py"),
        ("esm.utils.generation", "generation.py"),
        ("esm.models.esm3", "esm3.py"),
    ):
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(name)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
            raise
        setattr(parent, attribute, module)


_install_esm_backend()
del _install_esm_backend

from .loaders.model_loader import load_solv_encoder, load_thermo
from .hardware import Hardware_Setting
from .predict import predict
from .generate.generate import de_novo, gen_embed
