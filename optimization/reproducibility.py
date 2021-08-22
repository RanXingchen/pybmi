import numpy as np
import torch
import os


def fix_seed(seed: int):
    """
    Fix seed for package NUMPY, and TORCH.
    Do not fix RANDOM seed here, in case the random used for search hyperparameters.
    
    Parameters
    ----------
    seed: int
        The seed number.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
