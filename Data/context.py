import torch


class Context():
    def __init__(self, n, bidirectional=True):
        """
        Calculate the context frame of input and target.

        Parameters
        ----------
        n : int
            The number of context for current frame.
        bidirectional : bool, optional
            The sides of context, if set to True, context include
            n frames of history and n frames of future, that makes
            the number of features of current step is 2 * n + 1.
        """
        self.n = n
        self.bidirectional = bidirectional

    def apply(self, x, y=None):
        """
        Apply input and target data to compute the context.

        Parameters
        ----------
        x : ndarray
            Input data which will add context. Shape: (T, N), where T is
            the length of the input and N is the features.
        y : ndarray, optional
            The corresponding target data of x. Shape: (T, M), where T is
            the length of the target which should be equal to length of x,
            and M is the features of y.
        """
        if self.n != 0:
            n_left = -self.n
            n_right = self.n if self.bidirectional else 0

            shifted = [x[-n_left + i:-(n_right - i)]
                       for i in range(n_left, n_right)]
            shifted.append(x[n_right - n_left:])
            x = torch.cat(shifted, dim=-1)
            if y is not None:
                y = y[self.n:self.n + x.shape[0] + 1]
        return x, y

    def inverse_apply(self, y):
        """
        Inverse apply the context, make sure the y has same length with
        original one.

        y : ndarray
            The target data after applied context. Shape: (T, M - n) if
            bidirectional is False, otherwise Shape (T, M - 2 * n).
        """
        if self.n != 0:
            # Use the first and end element to fill the lost sequences.
            fill_beg = y[:1]
            fill_end = y[-1:] if self.bidirectional \
                else torch.tensor([], device=y.device)
            for i in range(self.n):
                y = torch.cat((fill_beg, y, fill_end), dim=0)
        return y
