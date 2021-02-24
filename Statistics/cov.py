import torch
import numpy as np

from Utils.utils import check_params


def cov(x, y=None, biased=False, nanflag='includenan'):
    """
    Pytorch version of covariance matrix.

    Parameters
    ----------
    x : {ndarray, Tensor}
        If X is a vector, returns the variance. For matrices, where
        each row is an observation, and each column a variable,
        COV(X) is the covariance matrix. DIAG(COV(X)) is a vector of
        variances for each column, and SQRT(DIAG(COV(X))) is a
        vector of standard deviations.
    y : {ndarray, Tensor}, optional
        When Y is provided, it need have same number of elements as X,
        COV(X, Y) is equivalent to
        COV(torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)).
    biased : bool, optional
        When biased is True, COV(X) or COV(X, Y) normalizes by N and
        produces the second moment matrix of the observations about
        their mean. When biased is False, COV(X) or COV(X, Y)
        normalizes by N - 1 if N > 1, where N is the number of
        observations. This makes COV(X) the best unbiased estimate
        of the covariance matrix if the observations are from a
        normal distribution. For N=1, COV normalizes by N.
    nanfalg : str, optional
        NANFLAG specifies how NaN (Not-A-Number) values are treated.
        The default is 'includenan':
            'includenan'   - if the input contains NaN, the output
                             also contains NaN. Specifically, C(I, J)
                             is NaN if column I or J of X contains
                             NaN values.
            'omitrows'     - omit all rows of X that contain NaN values:
                                ind = all(~isnan(X), 2)
                                C = cov(X(ind, :))
            'partialrows'  - compute each element C(I, J) separately,
                             based only on the columns I and J of X.
                             Omit rows only if they contain NaN values
                             in column I or J of X. The resulting matrix
                             C may not be a positive definite.
                                ind = all(~isnan(X(:, [I J])))
                                Clocal = cov(X(ind, [I J]))
                                C(I, J) = Clocal(1, 2)

    Returns
    -------
    c : Tensor
        If X is a vector, returns the variance. For matrices, c is the
        covariance matrix.
    """
    # Convert numpy ndarray to torch tensor
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    if type(y) is np.ndarray:
        y = torch.from_numpy(y)

    # Treat all vectors of x like column vectors.
    if len(x.shape) == 1 or x.shape[0] == 1:
        x = x.reshape(-1, 1)

    if y is not None:
        # Treat all vectors of y like column vectors.
        if len(y.shape) == 1 or y.shape[0] == 1:
            y = y.reshape(-1, 1)

        assert len(y.shape) == 2, \
            f"The input dimension of y is {y.shape}, it must be 2D."
        assert x.numel() == y.numel(), "Number of elements of " \
            f"x and y mismatch, x: {x.numel()}, y: {y.numel()}"
        # !Note: when multi-device, both X and Y are cuda,
        # !say "cuda:0" and "cuda:1", this might be a wrong assertion.
        assert x.device == y.device, "Device of x and y mismatch. " \
            "x: " + str(x.device) + "y: " + str(y.device)

        # Convert two inputs to equivalent single input.
        x = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=-1)

    # Check the dimension of x, the dimension > 2D is not acceptable.
    assert len(x.shape) == 2, \
        f"The input dimension of x is {x.shape}, it must be 2D."

    # Check the input argument is validate or not.
    nanflag = check_params(
        nanflag, ['omitrows', 'partialrows', 'includenan'], nanflag
    )
    omitnan = (nanflag == 'omitrows') or (nanflag == 'partialrows')
    dopairwise = (nanflag == 'partialrows')

    if omitnan:
        xnan = torch.isnan(x)

        if torch.any(xnan):     # otherwise, just do standard cov
            if dopairwise:
                c = _apply_pairwise(x, biased)
                return c
            else:
                nanrows = torch.any(xnan, dim=1)
                x = x[~nanrows]

    m, n = x.shape

    # The unbiased estimator: divide by (m-1).  Can't do this
    # when m == 0 or 1. The biased estimator: divide by m.
    # when m == 0 => return NaNs, m == 1 => return zeros
    denom = m if biased or m <= 1 else m - 1

    # Remove mean
    xc = x - x.mean(dim=0)
    c = torch.matmul(xc.T, xc) / denom
    return c


def _apply_pairwise(x, biased):
    """
    Apply cov pairwise to columns of x, ignoring NaN entries
    """
    n = x.shape[1]

    # First fill in the diagonal:
    c = torch.diag(_localcov_elementwise(x, x, biased))

    # Now compute off-diagonal entries
    for j in range(1, n):
        x1 = x[:, [j]].repeat(1, j)
        x2 = x[:, :j].clone()

        # make x1, x2 have the same nan patterns
        x1[torch.isnan(x2)] = float('NaN')
        x2[torch.isnan(x[:, j]), :] = float('NaN')

        c[j, :j] = _localcov_elementwise(x1, x2, biased)
    return c + torch.tril(c, -1).T


def _localcov_elementwise(x, y, biased):
    """
    Return c(i) = cov of x(:, i) and y(:, i), for all i
    with no error checking and assuming NaNs are removed
    returns 1xn vector c. x, y must be of the same size,
    with identical nan patterns.
    """
    nr_notnan = torch.sum(~torch.isnan(x), dim=0)
    x_omitnan, y_omitnan = x.clone(), y.clone()
    x_omitnan[torch.isnan(x)] = 0
    y_omitnan[torch.isnan(y)] = 0
    xc = x - x_omitnan.sum(dim=0) / nr_notnan
    yc = y - y_omitnan.sum(dim=0) / nr_notnan

    if ~biased:
        denom = nr_notnan - 1
        denom[nr_notnan == 1] = 1
        denom[nr_notnan == 0] = 0
    else:
        denom = nr_notnan

    xy = xc.conj() * yc
    xy_omitnan = xy.clone()
    xy_omitnan[torch.isnan(xy)] = 0
    c = xy_omitnan.sum(dim=0) / denom

    # Don't omit NaNs caused by computation (not missing data)
    ind = torch.any(torch.isnan(xy) & ~torch.isnan(x), dim=0)
    c[ind] = float('NaN')
    return c
