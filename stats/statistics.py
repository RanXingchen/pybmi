import torch
import numpy as np

from scipy.stats import f


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


def wanova1(y: np.ndarray, g: np.ndarray):
    """
    Welch's alternative to one-way analysis of variance (ANOVA).

    The goal of wanova1 is to test whether the population means of data taken
    from k different groups are all equal. This test does not require the
    condition of homogeneity of variances be satisfied.

    Data should be given in a single vector y with groups specified by a
    corresponding vector of group labels g (e.g., numbers from 1 to k). This
    is the general form which does not impose any restriction on the number of
    data in each group or the group labels.

    Under the null of constant means, the Welch's test statistic F follows an
    F distribution with df1 and df2 degrees of freedom.

    Bibliography:
    [1] Welch (1951) On the Comparison of Several Mean Values: An
        Alternative Approach. Biometrika. 38(3/4):330-336
    [2] Tomarken and Serlin (1986) Comparison of ANOVA alternatives
        under variance heterogeneity and specific noncentrality
        structures. Psychological Bulletin. 99(1):90-99.

    Parameters
    ----------
    y : ndarray
        The data to perform Welch's ANOVA. Shape of y should be 1D.
    g : ndarray
        The labels specified the k groups. The length of g have to same with y.

    Note
    ----
    This code is followed by the wanova.m of Andrew Penn (2022):
    https://www.mathworks.com/matlabcentral/fileexchange/61661-wanova
    """
    assert y.ndim == 1 and g.ndim == 1, \
        "The input data and labels should be a vector."

    # Determine the number of groups.
    labels = np.unique(g)
    k = len(labels)

    # Obtain the size, mean, and variance for each group.
    n, m, v = [], [], []
    for label in labels:
        n.append(np.size(y[g == label]))
        m.append(np.mean(y[g == label]))
        v.append(np.var(y[g == label], ddof=1))
    n, m, v = np.asarray(n), np.asarray(m), np.asarray(v)

    # Computing the standard errors of the mean (SEM) for each group.
    sem = v / n
    # Get the reciprocal of the SEM.
    w = 1 / sem
    # Calculate the origin.
    ori = np.sum(w * m) / np.sum(w)
    # Calculate Welch's test F ratio.
    numerator = np.sum(w * (m - ori) ** 2) / (k - 1)
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1) *
                       np.sum((1 - w / np.sum(w)) ** 2 / (n - 1)))
    F = numerator / denominator

    # Calculate the degrees of freedom.
    df1 = k - 1
    df2 = 1 / (3 / (k ** 2 - 1) * np.sum((1 - w / np.sum(w)) ** 2 / (n - 1)))
    # Calculate the p-value.
    p = 1 - f.cdf(F, df1, df2)
    return p
