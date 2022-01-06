import numpy as np

from scipy.interpolate import interp1d, PchipInterpolator
from dtw import accelerated_dtw


class DTW():
    def compute(self, x: np.ndarray, y: np.ndarray):
        """
        Compute the DTW Wrap path.

        Parameters
        ----------
        x : ndarray
            Input sequence.
        y : ndarray
            Target sequence.
        """
        y_len = np.linspace(0, y.shape[0] - 1, y.shape[0])
        new_y_len = np.linspace(0, 1, x.shape[0]) * \
            (y.shape[0] - 1)

        pchip = PchipInterpolator(y_len, y, extrapolate=True)
        y = pchip(new_y_len)

        x, y = np.flipud(x), np.flipud(y)

        _, _, _, path = accelerated_dtw(
            x, y, dist='euclidean'
        )
        p, q = path[0], path[1]

        align_idx = np.asarray(
            [q[np.where(p >= i)[0][0]] 
             for i in range(y.shape[0])]
        )
        self.align_idx_norm = align_idx / (y.shape[0] - 1)

    def apply(self, target_len: int, x: np.ndarray):
        x_len = np.linspace(0, x.shape[0] - 1, x.shape[0])
        new_x_len = np.linspace(0, 1, target_len) * \
            (x.shape[0] - 1)

        pchip_1 = PchipInterpolator(
            x_len, x, extrapolate=True
        )
        x = np.flipud(pchip_1(new_x_len))

        align_idx = self.align_idx_norm * (target_len - 1)
        align_idx_len = np.linspace(
            0, align_idx.shape[0] - 1, align_idx.shape[0]
        )
        new_align_idx_len = np.linspace(
            0, align_idx.shape[0] - 1, target_len
        )

        interp = interp1d(align_idx_len, align_idx, axis=0)
        new_align_idx = interp(new_align_idx_len)

        x_len = np.linspace(0, x.shape[0] - 1, x.shape[0])
        pchip_2 = PchipInterpolator(
            x_len, x, extrapolate=True
        )
        y = np.flipud(pchip_2(new_align_idx))
        return y
