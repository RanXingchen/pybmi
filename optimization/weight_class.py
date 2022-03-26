import torch
import numpy as np

from ..utils.utils import check_params


def wclass(label, method='weight', dtype=torch.float32, device='cpu'):
    """
    Computing the weight of each class for unbalanced dataset.

    Paraemters
    ----------
    label : Tensor or ndarray
        The label of all samples class.
    method : str, optional
        The computing type of weight of each class. It can be
        'weight' or 'proportion'. Default: 'weight'.
    """
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    elif not isinstance(label, np.ndarray):
        raise('Error type of label!')

    method = check_params(method, ['weight', 'proportion'], 'method')

    label = label.astype(np.int64)
    C = int(max(label) + 1)
    # The ideal weight of each class.
    w = torch.ones((C,), device=device) / C
    # If the sample of each class is unbalance in the dataset, this parameter
    # used to balance the class by making the optimizer more focus on the class
    # which had fewer samples.
    p = torch.from_numpy(np.bincount(label) / len(label)).to(device)
    w = w / p if method == 'weight' else p
    return w.type(dtype)
