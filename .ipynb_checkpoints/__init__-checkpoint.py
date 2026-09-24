# Copyright (c) 2026 Ernest Ngai Hei Ho
# Email: hoernesta@gmail.com
# Licensed under the MIT License.
__version__="0.1.0"
from .loaders.model_loader import load_solv_encoder, load_thermo
from .hardware import Hardware_Setting
# from .esm_gen import gen_embed
from .predict import predict
from .generate.generate import de_novo, gen_embed


## Replace esm3 script during import
import shutil
from importlib.resources import files

_source_folder="replace_esm3"
_package_path = files("heiwa")
shutil.copy(
    _package_path.joinpath(_source_folder, "esm3.py"), 
    _package_path.parent.joinpath("esm/models/esm3.py")
)
shutil.copy(
    _package_path.joinpath(_source_folder, "generation.py"), 
    _package_path.parent.joinpath("esm/utils/generation.py")
)
# shutil.copy(
#     _package_path.joinpath(_source_folder, "sequence_tokenizer.py"), 
#     _package_path.parent.joinpath("esm/tokenization/sequence_tokenizer.py")
# )
# shutil.copy(
#     _package_path.joinpath(_source_folder, "encoding.py"), 
#     _package_path.parent.joinpath("esm/utils/encoding.py")
# )
shutil.copy(
    _package_path.joinpath(_source_folder, "transformer_stack.py"), 
    _package_path.parent.joinpath("esm/layers/transformer_stack.py")
)
shutil.copy(
    _package_path.joinpath(_source_folder, "blocks.py"), 
    _package_path.parent.joinpath("esm/layers/blocks.py")
)

del _source_folder, _package_path
