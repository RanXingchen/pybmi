import torch
import numpy as np


def wclass(label, device='cpu'):
    """
    Computing the weight of each class for unbalanced dataset.

    Paraemters
    ----------
    label : Tensor or ndarray
        The label of all samples class.
    """
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    elif not isinstance(label, np.ndarray):
        raise('Error type of label!')

    label = label.astype(np.int64)
    C = int(max(label) + 1)
    # The ideal weight of each class.
    w = torch.ones((C,), device=device) / C
    # If the sample of each class is unbalance in the dataset, this parameter
    # used to balance the class by making the optimizer more focus on the class
    # which had fewer samples.
    proportion = np.bincount(label) / len(label)
    w /= torch.from_numpy(proportion).to(device)
    return w
