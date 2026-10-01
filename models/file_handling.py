import os
import subprocess
import shutil
from pathlib import Path

        
def compress(filepath, num_cpu=8):
    """Compress file with parallel gzip (replaces original with .gz)"""
    if os.path.exists(filepath) and not os.path.exists(f"{filepath}.gz"):
        cmd = f"pigz -p {num_cpu} {filepath}"
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath)
        # Now you have filepath.gz, and filepath is gone

def decompress(filepath, num_cpu=8):
    """Decompress file with parallel gzip (replaces .gz with original)"""
    if not os.path.exists(filepath) and os.path.exists(f"{filepath}.gz"):
        cmd = f"pigz -d -p {num_cpu} {filepath}.gz"
        subprocess.run(cmd, shell=True, check=True)
        # Now you have filepath, and filepath.gz is gone
        compressed_path = f"{filepath}.gz"
        if os.path.exists(compressed_path):
            os.remove(compressed_path)


def load_weights(path, device):
    """Read packaged weights without modifying the installation or requiring pigz."""
    import gzip
    import torch
    path = Path(path)
    if path.exists():
        return torch.load(path, map_location=device, weights_only=True)
    with gzip.open(str(path) + ".gz", "rb") as stream:
        return torch.load(stream, map_location=device, weights_only=True)
