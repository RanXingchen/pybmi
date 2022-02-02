import torch
import numpy as np

from tqdm import tqdm


def kfold(trainx, trainy, nfold, func, *arg):
    """
    Running specified function using k-fold cross validation.

    Parameters
    ----------
    trainx : ndarray or tensor
        The input training data. According the NFOLD, trainx will
        be split to training set and validation set. If NFOLD is
        0 or 1, the validation set will be None.
    trainy : ndarray or tensor
        The target training data. According the NFOLD, trg will
        be split to training set and validation set, where the index
        have same order as the inp. If NFOLD is 0 or 1, the
        validation set will be None.
    nfold : int
        The number of fold used to do cross validation. If nfold is
        0 or 1, no cross validation was used.
    funcs : function handle
        The function used to process the inp and trg data. Note
        that the input argument of funcs must satisfied
        funcs(trainx_, trainy_, evalx, evaly, testx, testy, *args)
        Where (trainx_, eval_x) and (trainy_, evaly) are splited
        from trainx and trainy according the NFOLD, and *args are
        other needed parameters provided.
    args : tuple
        Other input parameters that funcs needed.

    Returns
    -------
    losses : ndarray
        The losses of the k-fold cross validation. What type of loss
        are determined by the FUNCS.
    """
    losses = []

    if nfold == 0 or nfold == 1:
        # Number of fold is 0, the validation data is None.
        _, loss = func(trainx, trainy, None, None, *arg)
        losses.append(loss.item())
    else:
        # The number of validation data points in each fold
        len_eval_inp = trainx.shape[0] // nfold
        len_eval_trg = trainy.shape[0] // nfold
        for k in tqdm(range(nfold), desc=f'Running {nfold} fold cv'):
            # Mark the index of validation data to True to split them.
            idx_eval_inp = np.zeros(trainx.shape[0], dtype=np.bool)
            idx_eval_inp[k * len_eval_inp:(k + 1) * len_eval_inp] = True
            idx_eval_trg = np.zeros(trainy.shape[0], dtype=np.bool)
            idx_eval_trg[k * len_eval_trg:(k + 1) * len_eval_trg] = True
            # Split the training data and validation data.
            tinp, einp = trainx[~idx_eval_inp], trainx[idx_eval_inp]
            ttrg, etrg = trainy[~idx_eval_trg], trainy[idx_eval_trg]
            # Decoding the data of current fold.
            _, loss = func(tinp, ttrg, einp, etrg, *arg)
            # Store the criterion value.
            losses.append(loss.item())
            # Empty the stored cache for CUDA
            torch.cuda.empty_cache()
    return np.array(losses)
