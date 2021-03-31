import torch


def beta(z, w):
    """
    Beta function.

    BETA(Z,W) computes the beta function for corresponding
    elements of Z and W.  The beta function is defined as:
        beta(z,w) = integral from 0 to 1 of t.^(z-1) .* (1-t).^(w-1) dt.

    Parameters
    ----------
    z : float or Tensor
        Real and nonnegative float or Tensor data.
    w : float or Tensor
        Real and nonnegative float or Tensor data, the shape
        of w should be same with z if both are Tensor, or either
        can be float scalar.
    """
    return torch.exp(lbeta(z, w))


def ibeta(x, z, w):
    """
    Incomplete beta function.

    Y = BETAINC(X,Z,W) computes the incomplete beta function for corresponding
    elements of X, Z, and W.

    Parameters
    ----------
    x : tensor
        The elements of x must be in the closed interval [0, 1].
    z : float or Tensor
        Real and nonnegative float or Tensor data.
    w : float or Tensor
        Real and nonnegative float or Tensor data, the shape
        of w should be same with z if both are Tensor, or either
        can be float scalar.
    """
    print('not implement yet!')


def lbeta(z, w):
    """
    Logarithm of beta function.
    This function computes the natural logarithm of the beta
    function for corresponding elements of Z and W.
    LBETA is defined as:

        LBETA = LOG(BETA(Z,W))

    and is obtained without computing BETA(Z, W). Since the beta
    function can range over very large or very small values, its
    logarithm is sometimes more useful.

    Parameters
    ----------
    z : float or Tensor
        Real and nonnegative float or Tensor data.
    w : float or Tensor
        Real and nonnegative float or Tensor data, the shape
        of w should be same with z if both are Tensor, or either
        can be float scalar.
    """
    # Check the data validation
    if isinstance(z, (int, float)):
        z = torch.tensor(z, dtype=torch.float)
    if isinstance(w, (int, float)):
        w = torch.tensor(w, dtype=torch.float)

    return z.lgamma() + w.lgamma() - (z + w).lgamma()
