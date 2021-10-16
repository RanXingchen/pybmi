import torch
import numpy as np

from typing import Optional
from torch import Tensor


class SequencePreprocessor():
    """
    Preprocess the sequence data in the form of DataLoader.
    Including:
        1. Do delay for the input (neural) data and
           target (motion) data.
        2. Do Z-score for the dataset.
        3. Computing velocity for the target data.

    Parameters
    ----------
    n_delay : int.
        Number of bins to delay.
    target : str, optional.
        The target type of the motion data. 'p' means
        position, 'v' means velocity. If used 'pv', then
        both position and velocity used for the target.
    """
    def __init__(self, n_delay, target='pv'):
        # Delay
        self.n_delay = n_delay
        self.target = target
        self.fitted = False

    def fit(self, x: Tensor, y: Tensor,
            normalize_mask_x: Tensor = None,
            normalize_mask_y: Tensor = None):
        """
        Fit the training data to computing the needed parameters.

        Parameters
        ----------
        x : Tensor
            Input data.
        y : Tensor
            Target data.
        normalize_mask_x : Tensor
            This mask used to mask out some features that do not need
            to be normalized. False means will be normalized, True
            means will be ignored. The length of this mask should be
            same with the input features.
        normalize_mask_y : Tensor
            Same useage with NORMALIZE_MASK_X. The length of this
            mask should be same with the target features.
        """
        # * 1. Computing Z-score parameters.
        self.X_SS = StandardScaler().fit(x)
        self.Y_SS = StandardScaler().fit(y)
        # Set the mean of masked index to 0, and std of masked index
        # to 1.
        if normalize_mask_x is not None:
            self.X_SS.mean_[normalize_mask_x] = 0
            self.X_SS.scale_[normalize_mask_x] = 1
        if normalize_mask_y is not None:
            self.Y_SS.mean_[normalize_mask_y] = 0
            self.Y_SS.scale_[normalize_mask_y] = 1
        # * 2. Initializing the Delay.
        self.delay = Delay(self.n_delay)

        self.fitted = True
        return self

    def apply(self, x: Tensor, y: Tensor):
        if self.fitted:
            # Z-score normalization for the input and target.
            x = self.X_SS.apply(x)
            y = self.Y_SS.apply(y)
            # Apply delay.
            x, y = self.delay.apply(x, y)
            # Record the first position of y in case need to
            # compute position from velocity.
            self.p0 = y[:, :1]
            # Computing the specified target type.
            # Note the y is default 'p', so no need to
            # process 'p' case.
            if 'v' in self.target:
                # Velocity is needed for the target.
                v = torch.zeros_like(y, device=y.device)
                v[:, :-1] = y[:, 1:] - y[:, :-1]
                y = v if self.target == 'v' else \
                    torch.cat((y, v), dim=-1)
        return x, y

    def inverse_apply(self, y: Tensor, pv_w=[0, 1]):
        if self.fitted:
            # According target type to get position of y.
            if 'v' in self.target:
                p0 = self.p0.repeat(1, y.size(0) - 1, 1)
                if self.target == 'v':
                    v = y
                    y[:, 1:] = \
                        p0 + torch.cumsum(v[:, :-1], dim=0)
                elif self.target == 'pv':
                    i = y.size(-1) // 2
                    v = y[:, :, i:]
                    y = y[:, :, :i]
                    # Weigthed sum p and v according the
                    # PV_RATIO.
                    p = p0 + torch.cumsum(v[:, :-1], dim=0)
                    y[:, 1:] = pv_w[0] * y[:, 1:] + \
                        pv_w[1] * p
                y[0] = self.p0
            # Inverse apply delay and z-score.
            y = self.delay.inverse_apply(y)
            y = self.Y_SS.inverse_apply(y)
        return y


class Delay():
    """
    Computing delay for the data.

    Parameters
    ----------
    n : int
        The number of bins will be delay. If N greater than
        0, means the target (motion) data delayed N bins for
        the input (neural) data. If N smaller than 0,
        means the input (neural) data delayed N bins for the
        target (motion) data.
    """
    def __init__(self, n: int):
        self.n_delay = n

    def apply(self, x : Optional[Tensor],
              y : Optional[Tensor]) -> tuple:
        """
        Apply the delay.

        Parameters
        ----------
        x : Tensor or ndarray or list.
            The input data. Shape: [length, features]
        y : Tensor or ndarray or list.
            The target data. Shape: [length, features]

        Returns
        -------
        The x and y after applied the delay, the length of
        x and y becomes to length - N.
        """
        if self.n_delay > 0:
            x = x[:-self.n_delay]
            y = y[self.n_delay:]
        elif self.n_delay < 0:
            x = x[-self.n_delay:]
            y = y[:self.n_delay]
        return x, y

    def inverse_apply(self, y : Optional[Tensor]):
        """
        Restor the length of data which after applied delay
        to original. Note the restoration is simple replicate
        the endpoint.

        Parameters
        ----------
        y : Tensor.
            The target data after applied delay.
        """
        if self.n_delay > 0:
            # N greater than 0, the target data lack the
            # start N time bins.
            fill = y[:1]

            for _ in range(self.n_delay):
                if isinstance(y, np.ndarray):
                    y = np.concatenate((fill, y))
                elif isinstance(y, Tensor):
                    y = torch.cat((fill, y))
                elif isinstance (y, list):
                    y.insert(0, fill)
                else:
                    raise Exception('Unknown data type.')
        elif self.n_delay < 0:
            # N samller than 0, the target data lack the
            # end N time bins.
            fill = y[-1:]

            for _ in range(-self.n_delay):
                if isinstance(y, np.ndarray):
                    y = np.concatenate((y, fill))
                elif isinstance(y, Tensor):
                    y = torch.cat((y, fill))
                elif isinstance (y, list):
                    y.append(fill)
                else:
                    raise Exception('Unknown data type.')
        return y


class StandardScaler():
    """
    Z-score normalization.
    """
    def __init__(self):
        self.fitted = False

    def fit(self, x: Tensor):
        self.scale_, self.mean_ = torch.std_mean(x, dim=0)
        self.fitted = True
        return self

    def apply(self, x: Tensor):
        if self.fitted:
            x = (x - self.mean_) / self.scale_
            x = torch.where(
                torch.isnan(x), torch.full_like(x, 0), x
            )
        return x

    def inverse_apply(self, x: Tensor):
        if self.fitted:
            x = x * self.scale_ + self.mean_
        return x
