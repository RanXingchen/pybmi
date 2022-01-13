import numpy as np
import torch
import os


def fix_seed(seed: int):
    """
    Fix seed for package NUMPY, and TORCH.
    Do not fix RANDOM seed here, in case the random used to search
    hyperparameters.

    Parameters
    ----------
    seed: int
        The seed number.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
