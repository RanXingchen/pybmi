import numpy as np


def integ2d(mat, x=None, y=None):
    """
    INTEG2D Approximate 2-D integral.
    This function approximates the 2-D integral of
    matrix MAT according to abscissa X and ordinate Y.

    Parameters
    ----------
    mat : ndarray
        MxN matrix to be integrated.
    x : ndarray
        N-row-vector indicating the abscissa integration path.
        Default: 1:N
    y : ndarray
        M-column-vector indicating the ordinate integration path.
        Default: 1:M

    Ruturns
    -------
    som : ndarray
        Result of integration.
    """
    m, n = mat.shape

    if x is None:
        x = np.linspace(1, n, n)
        x = np.reshape(x, (1, n))
    if y is None:
        y = np.linspace(1, m, m)
        y = np.reshape(y, (m, 1))

    assert len(x.shape) == 2 and len(y.shape) == 2, \
        "X and Y must have only two dimension."

    xrow, xcol = x.shape
    yrow, ycol = y.shape
    assert xrow == 1, "X must be a row-vector."
    assert ycol == 1, "Y must be a column-vector."
    assert n == xcol, "MAT must have as many columns as X."
    assert m == yrow, "MAT must have as many rows as Y"

    mat = (np.sum(mat, axis=1, keepdims=True) - mat[:, :1] / 2 -
           mat[:, -1:] / 2) * (x[:, 1] - x[:, 0])
    dmat = mat[:-1] + mat[1:]
    dy = (y[1:] - y[:-1]) / 2
    som = np.sum(dmat * dy, axis=0)
    return som
