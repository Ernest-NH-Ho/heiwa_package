import numpy as np
from heiwa.utils import asset_path

logit_ranks = np.load(asset_path.joinpath("logit_ranks.npy"))
rmsf_ranks = np.load(asset_path.joinpath("rmsf_ranks.npy"))

def convert_logit_to_rmsf(logits):
    ranks = np.sum(logits > np.reshape(logit_ranks, (1, -1)), axis=-1)
    ranks = len(rmsf_ranks) - ranks
    ranks = np.where(ranks>=len(rmsf_ranks), len(rmsf_ranks)-1, ranks)
    return rmsf_ranks[ranks]