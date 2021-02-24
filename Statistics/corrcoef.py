import torch
import numpy as np
import math

from Utils.utils import check_params
from SpecialFunc.ibeta import ibeta
from .cov import cov


# TODO: implement p-value calculate if pytorch update to 1.8.0

def corrcoef(X, Y=None, alpha=0.05, rows='all'):
    """
    Calculate correlation coefficients of column vectors X and Y
    or matrix X itself.

    Parameters
    ----------
    X : ndarray
        When Y is None, the input X should be a matrix, where each
        row is an observation and each column is a variable. While
        Y is not None, both X and Y should be column vectors.
    Y : ndarray, optional
        When Y is provided, X and Y need to be column vectors,
        otherwise, this function converts X and Y to it. CORRCOEF(X, Y)
        is equivalent to CORRCOEF(torch.cat([X, Y], dim=1)).
    alpha : float, optional
        A number between 0 and 1 to specify a confidence level of
        100*(1-ALPHA). Default is 0.05 for 95% confidence intervals.
    rows : str, optional
        Either 'all' (default) to use all rows, 'complete' to use
        rows with no NaN values, or 'pairwise' to compute R[i,j]
        using rows with no NaN values in column i or j. The 'pairwise'
        option potentially uses different sets of rows to compute
        different elements of R, and can produce a matrix that is
        indefinite.

    Returns
    -------
    r : ndarray
        The calculated correlation coefficient of each column of X and Y.
    p : ndarray
        A matrix of p-values for testing the hypothesis of no correlation.
        Each p-value is the probability of getting a correlation as large
        as the observed value by random chance, when the true correlation
        is zero. If P[i,j] is small, say less than 0.05, then the correlation
        R[i,j] is significant. In the earily version of pytorch, <1.8.0,
        the p value is not calculated, it alwalys return 0.

    Notes
    -----
    The p-value is computed by transforming the correlation to create a t
    statistic having N-2 degrees of freedom, where N is the number of rows
    of X.

    Examples
    --------
    Generate random data having correlation between column 4 and the other
    columns.
    >>> x = torch.randn(30, 4)
    >>> x[:, -1] = x.sum(dim=-1)
    >>> r, p = corrcoef(x)
    >>> idx = torch.where(p < 0.05)
    >>> print(idx)
    """
    # Convert numpy ndarray to torch tensor
    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
    if type(Y) is np.ndarray:
        Y = torch.from_numpy(Y)

    # Treat all vectors of X like column vectors.
    if len(X.shape) == 1 or X.shape[0] == 1:
        X = X.reshape(-1, 1)

    if Y is not None:
        # Treat all vectors of Y like column vectors.
        if len(Y.shape) == 1 or Y.shape[0] == 1:
            Y = Y.reshape(-1, 1)

        assert len(Y.shape) == 2, \
            f"The input dimension of Y is {Y.shape}, it must be 2D."
        assert X.numel() == Y.numel(), "Number of elements of " \
            f"X and Y mismatch, X: {X.numel()}, Y: {Y.numel()}"
        # !Note: when multi-device, both X and Y are cuda,
        # !say "cuda:0" and "cuda:1", this might be a wrong assertion.
        assert X.device == Y.device, "Device of X and Y mismatch. " \
            "X: " + str(X.device) + "Y: " + str(Y.device)
        # Convert two inputs to equivalent single input.
        X = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=-1)

    assert len(X.shape) == 2, \
        f"The input dimension of X is {X.shape}, it must be 2D."

    # Check the input parameters
    rows = check_params(rows, ['all', 'complete', 'pairwise'], 'rows')

    n, m = X.shape

    # Compute correlations.
    t = torch.isnan(X)
    removemissing = torch.any(t)
    if rows == 'all' or ~removemissing:
        r, n = _corr(X)
    elif rows == 'complete':
        # Remove observations with missing values.
        X = X[~torch.any(t, dim=1)]
        if X.shape[0] == 1:
            # X is now a row vector, but wasn't a row vector on input.
            r = torch.Tensor(X.shape[1], X.shape[1]).\
                fill_(float('NaN')).to(X.device)
            n = 1
        else:
            r, n = _corr(X)
    else:
        # Compute correlation for each pair.
        r, n = _corrpairwise(X)

    # Compute p-value.
    # ! Since the igamma is not implement in pytorch 1.7.1, we comment
    # ! the p-value part.
    # lowerhalf = torch.tril(torch.ones(m, m, device=X.device), -1) > 0
    # rv = r[lowerhalf]
    # nv = n[lowerhalf] if len(n) > 1 else n

    # # tstat = +/-Inf and p = 0 if abs(r) == 1, NaN if r == NaN.
    # tstat = rv * ((nv - 2) / (1 - rv ** 2)).sqrt()
    # p = torch.zeros_like(r)
    # p[lowerhalf] = 2 * _tpvalue(-tstat.abs(), nv - 2)
    # # Preserve NaNs on diag.
    # p = p + p.T + r.diag().diag()

    # * For temporarily
    p = 0

    return r, p


def _corr(x):
    """
    Compute correlation matrix without error checking.
    """
    n, m = x.shape
    r = cov(x)
    # sqrt first to avoid under/overflow
    d = r.diag().sqrt().reshape(-1, 1)
    r = r / d / d.T
    # Fix up possible round-off problems, while preserving NaN:
    # put exact 1 on the diagonal, and limit off-diag to [-1,1].
    r = (r + r.T) / 2
    t = r.abs() > 1
    r[t] = r[t].sign()
    r = r - r.diag().diag() + r.diag().sign().diag()
    return r, n


def _corrpairwise(x):
    """
    Apply corrcoef pairwise to columns of x, ignoring NaN entries
    """
    n = x.shape[1]

    # First fill in the diagonal:
    # Fix up possible round-off problems, while preserving NaN: put exact 1 on
    # the diagonal, unless computation results in NaN.
    c = _localcorrcoef_elementwise(x, x)[0].sign().diag()
    nr_notnan = torch.zeros((n, n), dtype=x.dtype, device=x.device)

    # Now compute off-diagonal entries
    for j in range(1, n):
        x1 = x[:, [j]].repeat(1, j)
        x2 = x[:, :j].clone()

        # make x1, x2 have the same NaN patterns
        x1[torch.isnan(x2)] = float('NaN')
        x2[torch.isnan(x[:, j]), :] = float('NaN')

        c[j, :j], nr_notnan[j, :j] = _localcorrcoef_elementwise(x1, x2)
    c += torch.tril(c, -1).T
    nr_notnan += torch.tril(nr_notnan, -1).T

    # Fix up possible round-off problems: limit off-diag to [-1,1].
    t = c.abs() > 1
    c[t] = c[t].sign()
    return c, nr_notnan


def _localcorrcoef_elementwise(x, y):
    """
    Return c(i) = corrcoef of x(:, i) and y(:, i), for all i
    with no error checking and assuming NaNs are removed
    returns 1xn vector c. x, y must be of the same size, with
    identical NaN patterns
    """
    nr_notnan = torch.sum(~torch.isnan(x), dim=0)
    x_omitnan, y_omitnan = x.clone(), y.clone()
    x_omitnan[torch.isnan(x)] = 0
    y_omitnan[torch.isnan(y)] = 0
    xc = x - x_omitnan.sum(dim=0) / nr_notnan
    yc = y - y_omitnan.sum(dim=0) / nr_notnan

    denom = nr_notnan - 1
    denom[nr_notnan == 1] = 1
    denom[nr_notnan == 0] = 0

    xy = xc.conj() * yc
    txy = torch.isnan(xy)
    xy_omitnan = xy.clone()
    xy_omitnan[txy] = 0
    cxy = xy_omitnan.sum(dim=0) / denom
    xx = xc.conj() * xc
    txx = torch.isnan(xx)
    xx_omitnan = xx.clone()
    xx_omitnan[txx] = 0
    cxx = xx_omitnan.sum(dim=0) / denom
    yy = yc.conj() * yc
    tyy = torch.isnan(yy)
    yy_omitnan = yy.clone()
    yy_omitnan[tyy] = 0
    cyy = yy_omitnan.sum(dim=0) / denom

    c = cxy / cxx.sqrt() / cyy.sqrt()

    # Don't omit NaNs caused by computation (not missing data)
    ind = torch.any((txy | txx | tyy) & ~torch.isnan(x), dim=0)
    c[ind] = float('NaN')
    return c, nr_notnan


def _tpvalue(x, v):
    """
    Compute p-value for t statistic.
    """
    normcutoff = 1e7
    if len(x) != 1 and len(v) == 1:
        v = v.repeat(x.shape)

    # Initialize P
    p = torch.Tensor(x.shape).fill_(float('NaN'))
    # v == NaN ==> (0 < v) == false
    nans = torch.isnan(x) | ~(0 < v)

    # First compute F(-|x|).
    #
    # Cauchy distribution.  See Devroye pages 29 and 450.
    cauchy = v == 1
    p[cauchy] = 0.5 + x[cauchy].atan() / math.pi

    # Normal Approximation.
    normal = v > normcutoff
    p[normal] = 0.5 * (-x[normal] / torch.sqrt(2)).erfc()

    # See Abramowitz and Stegun, formulas 26.5.27 and 26.7.1.
    gen = ~(cauchy | normal | nans)
    # ! ibeta is not implement before 1.8.0
    p[gen] = ibeta(v[gen] / (v[gen] + x[gen] ** 2), v[gen] / 2, 0.5) / 2

    # Adjust for x>0.  Right now p<0.5, so this is numerically safe.
    reflect = gen & (x > 0)
    p[reflect] = 1 - p[reflect]

    # Make the result exact for the median.
    p[x == 0 & ~nans] = 0.5
    return p
