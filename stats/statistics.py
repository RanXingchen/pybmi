import torch
import numpy as np


def wnanmean(X, W=None, axis=0):
    """
    Weighted mean values ignoring NaN's.
    Computing mean for matrix X along specified dimension weighted
    by vector W.

    Parameters
    ----------
    X : torch tensor or ndarray
        Input array.
    W : torch tensor or ndarray, optional
        The weight multiply X that performed on the specified axis. If it is
        None, ones vector that has the same shape as X will be created.
    axis : int, optional.
        It specifies the axis of X along which to compute the weighted NaN
        mean. The default is 0.

    Returns
    -------
    M : torch tensor
        The weighted mean of X along the specified axis.
    """
    # Convert numpy ndarray to torch tensor
    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
    if type(W) is np.ndarray:
        W = torch.from_numpy(W)

    # Dimension of X
    D = X.ndim
    # Get the suitable shape of W can be used on later multiply.
    shape = np.ones(D, dtype=int).tolist()
    shape[axis] = X.shape[axis]
    # Create ones vector as W along the axis if W is None.
    if W is None:
        W = torch.ones(shape, dtype=X.dtype, device=X.device)

    # Make sure W has a appropriate shape
    assert W.numel() == shape[axis], \
        f'Wrong size of W: {W.numel()}, correct size should be {shape[axis]}.'
    W = torch.reshape(W, shape)
    W[torch.isnan(W)] = 0

    # Weigthed X and set NaN's to 0.
    _X = X * W
    nan_idx = torch.isnan(_X)
    _X[nan_idx] = 0

    # Sum the not NaN's value of W along the axis.
    N = torch.sum(~nan_idx * W, dim=axis)
    M = torch.sum(_X, dim=axis) / N
    return M


def wnanvar(X, W=None, bias=1, axis=0):
    """
    Weighted variance ignoring NaN's.

    Parameters
    ----------
    X : torch tensor or ndarray
        The input data which will be computed the variance along
        axis dimension.
    W : torch tensor or ndarray, optional
        The weight multiply X that performed on the specified axis. If it is
        None, ones vector that has the same shape as X will be created.
    bias : bool, optional
        If bias set to False, this function returns max likelihood estimate of
        variance along axis dimension of X weighted by vector W.
        If bias set to True, this function returns unbiased estimate of
        variance along axis dimension of X weighted by vector W.
    axis : int, optional
        The dimension of computing occored. Default: 0.

    Returns
    -------
    V : torch tensor
        The estimated variance of X.

    Notes
    -----
    There is a simple proof of weighted variance in the weighted_variance.md.
    """
    # Convert numpy ndarray to torch tensor
    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
    if type(W) is np.ndarray:
        W = torch.from_numpy(W)

    # Dimension of X
    D = X.ndim
    # Get the suitable shape of W can be used on later multiply.
    shape = np.ones(D, dtype=int).tolist()
    shape[axis] = X.shape[axis]
    # Create ones vector as W along the axis if W is None.
    if W is None:
        W = torch.ones(shape, dtype=X.dtype, device=X.device)

    # Make sure W has a appropriate shape
    assert W.numel() == shape[axis], \
        f'Wrong size of W: {W.numel()}, correct size should be {shape[axis]}.'
    W = torch.reshape(W, shape)
    W[torch.isnan(W)] = 0

    M = wnanmean(X, W, axis=axis)

    _X = (X - M) ** 2 * W
    nan_idx = torch.isnan(_X)
    _X[nan_idx] = 0

    # Sum the not NaN's value of W along the axis.
    N = torch.sum(~nan_idx * W, dim=axis)
    # Biased estimation of variance.
    V = torch.sum(_X, dim=axis) / N

    if _X.shape[axis] > 1 and bias:
        N2 = torch.sum(~nan_idx * W ** 2, dim=axis)
        # Unbiased estimation.
        # * The proof of this formula is under weighted_variance.md.
        V = torch.sum(_X, dim=axis) / (N - N2 / N)
    return V
