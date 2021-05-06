import torch
import numpy as np


def kfold(inp, trg, nfold, func, *arg):
    """
    Running specified function using k-fold cross validation.

    Parameters
    ----------
    inp : ndarray or tensor
        The input training data. According the NFOLD, inp will
        be split to training set and validation set.
    trg : ndarray or tensor
        The target training data. According the NFOLD, trg will
        be split to training set and validation set, where the index
        have same order as the inp.
    nfold : int
        The number of fold used to do cross validation. If nfold is
        0 or 1, no cross validation was used, the validation data is
        same as the training data.
    funcs : function handle
        The function used to process the inp and trg data.
    args : tuple
        Other input parameters that funcs needed.

    Returns
    -------
    mse : tensor
        The mean square error of the k-fold cross validation.
    """
    if nfold == 0 or nfold == 1:
        # Number of fold is 0, the validation data same as the training data.
        output = func(inp, trg, inp, trg, *arg)
        mse = ((output - trg) ** 2).mean(dim=0, keepdim=True)
        return mse
    else:
        # The number of validation data points in each fold
        len_eval = inp.shape[0] // nfold
        # mse is the list which stored the mean square error of every fold.
        mse = []
        for k in range(nfold):
            # Mark the index of validation data to True to split them.
            idx_eval = np.zeros(inp.shape[0], dtype=np.bool)
            idx_eval[k * len_eval:(k + 1) * len_eval] = True
            # Split the training data and validation data.
            tinp, einp = inp[~idx_eval], inp[idx_eval]
            ttrg, etrg = trg[~idx_eval], trg[idx_eval]
            # Decoding the data of current fold.
            output = func(tinp, ttrg, einp, etrg, *arg)
            # Compute the mean square error.
            mse.append(((output - etrg) ** 2).mean(dim=0, keepdim=True))
            # Empty the stored cache for CUDA
            torch.cuda.empty_cache()
        return torch.cat(mse, dim=0)
