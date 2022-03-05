import numpy as np
import torch

from pybmi.utils.utils import check_params


class Smooth():
    """
    Smooth data.

    Parameters
    ----------
    span : int, optional
        Smooths data Y using SPAN as the number of points used
        to compute each element of smoothed data. Default: 5.
    method : str, optional
        Smooths data Y with specified METHOD. The available methods
        are:
            'moving' - Moving average (default)

    Notes
    -----
    1. For the 'moving' methods, SPAN must be odd. If an even SPAN
       is specified, it is reduced by 1.
    2. If SPAN is greater than the length of Y, it is reduced to
       the length of Y.

    Examples
    --------
    >>> y = np.random.randn((1000, 5))
    >>> smoother = Smooth(span=5, method='moving')
    >>> smoothed_y = smoother.apply(y)
    """
    def __init__(self, span=5, method='moving'):
        self.span = span
        self.method = check_params(method, ['moving'], 'method')

    def apply(self, y):
        """
        Apply the smooth method to Y.

        Parameters
        ----------
        y : ndarray or Tensor
            The input data will be smoothed. The shape of Y should be
            [T, D], where T is the length, and D is the feature dimension.

        Returns
        -------
        smoothed : ndarray or Tensor
            The smoothed data of y.
        """
        if isinstance(y, torch.Tensor):
            istensor = True
            device = y.device
            dtype = y.dtype
            y = y.detach().cpu().numpy()
        else:
            istensor = False

        smoothed = y.copy()
        if self.method == 'moving':
            for n, i in enumerate(y.T):
                ii = np.convolve(
                    i, np.ones((self.span,)) / self.span, mode='valid'
                )
                r = np.arange(1, self.span - 1, 2)
                beg = np.cumsum(i[:self.span - 1])[::2] / r
                end = (np.cumsum(i[:-self.span:-1])[::2] / r)[::-1]
                smoothed[:, n] = np.concatenate((beg, ii, end))

        if istensor:
            smoothed = torch.tensor(smoothed, dtype=dtype).to(device)
        return smoothed
