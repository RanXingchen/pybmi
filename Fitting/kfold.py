import torch
import numpy as np


def kfold(trainx, trainy, testx, testy, nfold, func, *arg):
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
    testx : ndarray or tensor
        The input test data. Used to test the performance of the
        model after training and validation.
    testy : ndarray or tensor
        The target test data. Used to test the performance of the
        model after training and validation.
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
    epochs : ndarray
        The number of training epochs of the k-fold cross validation.
    losses : ndarray
        The losses of the k-fold cross validation. What type of loss
        are determined by the FUNCS.
    """
    epochs, losses = [], []

    if nfold == 0 or nfold == 1:
        # Number of fold is 0, the validation data is None.
        epoch, _, loss = func(trainx, trainy, None, None, testx, testy, *arg)
        epochs.append(epoch)
        losses.append(loss.item())
    else:
        # The number of validation data points in each fold
        len_eval = trainx.shape[0] // nfold
        for k in range(nfold):
            # Mark the index of validation data to True to split them.
            idx_eval = np.zeros(trainx.shape[0], dtype=np.bool)
            idx_eval[k * len_eval:(k + 1) * len_eval] = True
            # Split the training data and validation data.
            tinp, einp = trainx[~idx_eval], trainx[idx_eval]
            ttrg, etrg = trainy[~idx_eval], trainy[idx_eval]
            # Decoding the data of current fold.
            epoch, _, loss = func(tinp, ttrg, einp, etrg, testx, testy, *arg)
            # Store the criterion and epoches.
            epochs.append(epoch)
            losses.append(loss.item())
            # Empty the stored cache for CUDA
            torch.cuda.empty_cache()
    return np.array(epochs), np.array(losses)
