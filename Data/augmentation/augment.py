import torch
import math
import numpy as np
import scipy.io as scio
import torchvision.transforms.functional as TF
import warnings

from torch import Tensor
from pybmi.signals import DTW
from pybmi.utils import check_params
from typing import List


class DataAugment(object):
    """
    Data augmentation for the BCI input data.

    Parameters
    ----------
    p : float
        Probability of data augmentation in this iteration.
    w_array : int
        The number of channels of the array in each row.
    h_array : int
        The number of channels of the array in each column.
    n_bands : int
        Number of frequency bands of the neural data.
    methods : list
        The augmentation methods want to be used. It can be
        'jittering', 'scaling', 'permutation', 'rotation',
        'mixup'.
    """
    def __init__(self, p: float, w_array: int, h_array: int,
                 n_bands: int, methods: list):
        self.p = p
        self.w_array = w_array
        self.h_array = h_array
        self.n_bands = n_bands
        self.dtw = DTW()

        # Check the methods validation.
        choices = ['jittering', 'scaling', 'shuffle', 'permutation',
                   'rotation', 'padding', 'crop', 'mixup']
        self.methods = [check_params(m, choices, 'methods')
                        for m in methods]

    def __call__(self, X1: Tensor, Y1: Tensor = None,
                 X2: Tensor = None, Y2: Tensor = None):
        """
        X1 are must input, and Y1 only used when METHODs
        contain 'permutation', and X2 and Y2 only used
        when METHODs contain 'mixup'.
        """
        X, Y = X1, Y1
        if torch.rand(1) < self.p:
            # Only the probability of p for the
            # data augmentation.
            for method in self.methods:
                if method == 'jittering':
                    X = self.jittering(X)
                elif method == 'scaling':
                    X = self.scaling(X)
                elif method == 'shuffle':
                    X = self.shuffle(X)
                elif method == 'permutation':
                    assert Y is not None, \
                        "Method 'permutation' need target data to keep the " \
                        "correspondence between input and target data."
                    X, Y = self.permutation(X, Y)
                elif method == 'rotation':
                    X = self.rotation(X)
                elif method == 'padding':
                    X = self.padding(X)
                elif method == 'crop':
                    X = self.crop(X)
                elif method == 'mixup':
                    assert Y is not None, \
                        "Method 'permutation' need target data to keep the " \
                        "correspondence between input and target data."
                    assert X2 is not None, \
                        "Method 'mixup' need another sequence data to " \
                        "computation."
                    X, Y = self.mixup(X, Y, X2, Y2)
        return X, Y

    def jittering(self, x: Tensor):
        """
        Add N(0, 1) white noise to the input data.
        """
        # The total number of electrodes.
        N = self.w_array * self.h_array
        noise = torch.normal(
            mean=0, std=1, size=(N,),
            dtype=x.dtype, device=x.device
        ).repeat(1, self.n_bands)
        return x + noise

    def scaling(self, x: Tensor):
        """
        Multiply N(1, 1) white noise to the input data.
        """
        noise = torch.normal(
            mean=1, std=1, size=(x.shape[-1],),
            dtype=x.dtype, device=x.device
        )
        return x * noise

    def shuffle(self, x: Tensor):
        """
        Shuffle the order of features.
        """
        # The total number of electrodes.
        N = self.w_array * self.h_array
        # The number of frequency bands.
        F = self.n_bands

        x_ = x.reshape(-1, N, F)
        x_ = x_[:, torch.randperm(N), :]
        x_ = x_[:, :, torch.randperm(F)]
        return x_.reshape(-1, N * F)

    def permutation(self, x: Tensor, y: Tensor,
                    nPerm: int = 3, min_seg_length: int = 50):
        """
        Permutation the order of the sequence data.

        Parameters
        ----------
        nPerm : int, optional
            Number of the permutation segements.
        min_seg_length : int, optional
            The min segment length when do the permutation.
        """
        x_ = torch.zeros(x.shape, device=x.device)
        y_ = torch.zeros(y.shape, device=y.device)
        while 1:
            # SEGS contain the endpoint for each segment.
            segs = torch.zeros(nPerm + 1, dtype=torch.int)
            # The first element of SEGS is 0 and the last
            # element of SEGS is the length of x.
            segs[1:-1], _ = torch.sort(
                torch.randint(min_seg_length, x.shape[0] - min_seg_length,
                              (nPerm - 1,))
            )
            segs[-1] = x.shape[0]
            # Check if all segments satisfact to
            # min_seg_length. If not, redo the above step.
            if (segs[1:] - segs[:-1]).min() > min_seg_length:
                break
        # Permute x and y simutancely.
        pos = 0
        indices = torch.randperm(nPerm)
        for i in indices:
            temp = x[segs[i]:segs[i + 1]]
            x_[pos:pos + temp.shape[0]] = temp

            temp = y[segs[i]:segs[i + 1]]
            y_[pos:pos + temp.shape[0]] = temp

            pos += temp.shape[0]
        return x_, y_

    def rotation(self, x: Tensor, max_degree=15):
        """
        Rotate the Array of the electrodes.
        """
        F = self.n_bands
        # Obtain a random rotate angle.
        angle = torch.rand(1) * 2 * max_degree - max_degree
        if angle == 0:
            return x

        # The total number of electrodes.
        N = self.w_array * self.h_array
        assert N * F == x.size(-1), \
            'Error occored for the computation of neural '\
            'array channels.'
        # Width and height of the array.
        W = self.w_array
        H = self.h_array

        x = x.reshape(-1, N, F).permute(0, 2, 1).reshape(-1, F, H, W)
        x = TF.rotate(x, angle.item(), TF.InterpolationMode.BILINEAR)
        x = x.reshape(-1, F, N).permute(0, 2, 1).reshape(-1, N * F)
        return x

    def padding(self, x: Tensor, padding: List[int] = None, fill=0,
                mode='constant'):
        """
        Padding the given neural grid on all sides with the given
        "PAD" value.

        Parameters
        ----------
        x: Tensor
            The input neural data. It should be able to reshape to
            [..., N_BANDs, H_ARRAY, W_ARRAY], where the ... means
            the leading dimension.
        padding: int or sequence, optional
            Padding on each border. If a single int is provided
            this is used to pad all borders. If sequence of length
            2 is provided this is the padding on left/right and
            top/bottom respectively. If a sequence of length 4 is
            provided this is the padding for the left, top, right
            and bottom borders respectively. Default: None
        fill: int, optional
            The fill vaule for constant fill. This value is only
            used when MODE='constant'. Default: 0.
        mode: str, optional
            Type of padding. Should be one of 'constant', 'edge',
            'reflect', or 'symmetric'. Default: 'constant'.
            'constant' - Pads with a constant value, this value is
                         specified with fill.
            'edge'     - Pads with the last value at the edge of the
                         grid. If input a 5D torch Tensor, the last
                         3 dimensions will be padded instead of the
                         last 2.
            'reflect'  - Pads with reflection of grid without repeating
                         the last value on the edge. For example,
                         padding [1, 2, 3, 4] with 2 elements on both
                         sides in reflect mode will result in
                         [3, 2, 1, 2, 3, 4, 3, 2].
            'symmetric'- Pads with reflection of image repeating the
                         last value on the edge. For example, padding
                         [1, 2, 3, 4] with 2 elements on both sides in
                         symmetric mode will result in
                         [2, 1, 1, 2, 3, 4, 4, 3].
        """
        F = self.n_bands
        # The total number of electrodes.
        N = self.w_array * self.h_array
        assert N * F == x.size(-1), \
            'Error occored for the computation of neural '\
            'array channels.'
        # Width and height of the array.
        W, H = self.w_array, self.h_array

        # Specify the padding if it is none.
        if padding is None:
            max_padding = min(W // 2, H // 2) // 3
            # High value is excluded.
            padding = torch.randint(1, max_padding + 1, (1,)).item()

        x = x.reshape(-1, N, F).permute(0, 2, 1).reshape(-1, F, H, W)
        x = TF.resize(x, [H - padding * 2, W - padding * 2])
        x = TF.pad(x, padding, fill, mode)
        x = x.reshape(-1, F, N).permute(0, 2, 1).reshape(-1, N * F)
        return x

    def crop(self, x: Tensor, scale: List[float] = (0.08, 1.0),
             ratio: List[float] = (3. / 4., 4. / 3.)):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max).")

        F = self.n_bands
        # Width and height of the array.
        W, H = self.w_array, self.h_array
        # The total number of electrodes.
        N = W * H
        assert N * F == x.size(-1), \
            'Error occored for the computation of neural '\
            'array channels.'

        log_ratio = torch.log(torch.tensor(ratio))
        i, j = -1, -1
        for _ in range(10):
            target_area = N * torch.empty(1).uniform_(scale[0], scale[1])
            target_area = target_area.item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= W and 0 < h <= H:
                i = torch.randint(0, H - h + 1, size=(1,)).item()
                j = torch.randint(0, W - w + 1, size=(1,)).item()
                break

        if i == -1 and j == -1:
            # Fallback to central crop
            in_ratio = float(W) / float(H)
            if in_ratio < min(ratio):
                w = W
                h = int(round(w / min(ratio)))
            elif in_ratio > max(ratio):
                h = H
                w = int(round(h * max(ratio)))
            else:  # whole image
                w = W
                h = H
            i = (H - h) // 2
            j = (W - w) // 2

        # Random crop the grid according i, j, h, w.
        x = x.reshape(-1, N, F).permute(0, 2, 1).reshape(-1, F, H, W)
        x = TF.resized_crop(x, i, j, h, w, (H, W))
        x = x.reshape(-1, F, N).permute(0, 2, 1).reshape(-1, N * F)
        return x

    def mixup(self, x1, y1, x2, y2):
        """
        ! The mixup had some problems.
        """
        # Using DTW process additional sentence in order to have same length
        # with original one.
        x1_numpy = x1.cpu().numpy()
        x2_numpy = x2.cpu().numpy()
        y1_numpy = y1.cpu().numpy()
        y2_numpy = y2.cpu().numpy()

        self.dtw.compute(x1_numpy, x2_numpy)
        x_ = self.dtw.apply(x1.shape[0], x2_numpy)

        self.dtw.compute(y1_numpy, y2_numpy)
        y_ = self.dtw.apply(y1.shape[0], y2_numpy)

        lam = np.random.beta(0.4, 0.4)
        x = lam * x1_numpy + (1 - lam) * x_
        y = lam * y1_numpy + (1 - lam) * y_

        x = torch.tensor(
            x, dtype=torch.float, device=x1.device
        )
        y = torch.tensor(
            y, dtype=torch.float, device=y1.device
        )

        # Set mixed voiced value less than 0.5 to zero.
        f0, voice = y[:, -2], y[:, -1]
        voice[voice <= 0.5], voice[voice > 0.5] = 0, 1
        f0[~voice.bool()] = 0
        f0[f0 < 73.5], f0[f0 > 275.625] = 0, 275.625
        y[:, -2], y[:, -1] = f0, voice
        return x, y


if __name__ == '__main__':
    torch.manual_seed(42)

    augmentor = DataAugment(p=1, w_array=8, h_array=9,
                            n_bands=21, methods=['rotation'])
    data = torch.randn((1, 267, 1512), device='cuda')
    rotated = augmentor(data)[0]

    scio.savemat(
        'test_rotate.mat',
        {'data': data.squeeze(0).cpu().numpy(),
         'rotated': rotated.squeeze(0).cpu().numpy()}
    )
