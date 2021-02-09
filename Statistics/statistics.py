import numpy as np


def wnanmean(X, W=None, axis=0):
    """
    Weighted mean values ignoring NaN's.
    Computing mean for matrix X along specified dimension weighted
    by vector W.

    Parameters
    ----------
    X : array_like
        Input array.
    W : array_like, optional
        The weight multiply X that performed on the specified axis. If it is
        None, ones vector that has the same shape as X will be created.
    axis : int, optional.
        It specifies the axis of X along which to compute the weighted NaN
        mean. The default is 0.

    Returns
    -------
    M : array_like
        The weighted mean of X along the specified axis.
    """

    # Dimension of X
    D = len(X.shape)
    # Get the suitable shape of W can be used on later multiply.
    shape = np.ones(D, dtype=int)
    shape[axis] = X.shape[axis]
    # Create ones vector as W along the axis if W is None.
    if W is None:
        W = np.ones(shape, dtype=X.dtype)

    # Make sure W has a appropriate shape
    assert W.size == shape[axis], \
        f'Wrong size of W: {W.size}, the correct size should be {shape[axis]}.'
    W = np.reshape(W, shape)
    W[np.isnan(W)] = 0

    # Weigthed X and set NaN's to 0.
    _X = X * W
    nan_idx = np.isnan(_X)
    _X[nan_idx] = 0

    # Sum the not NaN's value of W along the axis.
    N = np.sum(~nan_idx * W, axis=axis)
    M = np.sum(_X, axis=axis) / N
    return M


def wnanvar(X, W=None, bias=1, axis=0):
    """
    Weighted variance ignoring NaN's.

    Parameters
    ----------
    X : array_like
        The input data which will be computed the variance along
        axis dimension.
    W : array_like, optional
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
    V : array_like
        The estimated variance of X.

    Notes
    -----
    There is a simple proof of weighted variance in the weighted_variance.md.
    """

    # Dimension of X
    D = len(X.shape)
    # Get the suitable shape of W can be used on later multiply.
    shape = np.ones(D, dtype=int)
    shape[axis] = X.shape[axis]
    # Create ones vector as W along the axis if W is None.
    if W is None:
        W = np.ones(shape, dtype=X.dtype)

    # Make sure W has a appropriate shape
    assert W.size == shape[axis], \
        f'Wrong size of W: {W.size}, the correct size should be {shape[axis]}.'
    W = np.reshape(W, shape)
    W[np.isnan(W)] = 0

    M = wnanmean(X, W, axis=axis)

    _X = (X - M) ** 2 * W
    nan_idx = np.isnan(_X)
    _X[nan_idx] = 0

    # Sum the not NaN's value of W along the axis.
    N = np.sum(~nan_idx * W, axis=axis)
    # Biased estimation of variance.
    V = np.sum(_X, axis=axis) / N

    if _X.shape[axis] > 1 and bias:
        N2 = np.sum(~nan_idx * W ** 2, axis=axis)
        # Unbiased estimation.
        # * The proof of this formula is under weighted_variance.md.
        V = np.sum(_X, axis=axis) / (N - N2 / N)
    return V
