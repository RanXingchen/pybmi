import torch
# import math
import numpy as np
# import scipy.io as scio
# import torchvision.transforms.functional as TF
# import warnings

from torch import Tensor
from torch.distributions.normal import Normal
# from typing import List


def mixup(x, y, alpha):
    """
    Parameters
    ----------
    x : ndarray
        The input data which will be mixed up. The shape of x should be
        [nsamples, time_points, nfeatures].
    y : ndarray
        The target data which will be mixed up.
        Shape: [nsamples, nclasses].
    alpha : float
        The parameter to control the mix propotion of two sample.
    """
    nsamples = x.shape[0]
    assert nsamples == y.shape[0], "The shape of x and y dismatch!"

    indices = np.arange(0, nsamples, dtype=np.int32)
    np.random.shuffle(indices)

    lam = np.random.beta(alpha, alpha)
    x = lam * x + (1 - lam) * x[indices]
    y_a, y_b = y, y[indices]
    return x, y_a, y_b, lam


def scaling(x: Tensor, mu=1.0, sigma=1.0) -> Tensor:
    """
    Multiply N(1, 1) white noise to the input data.

    Parameters
    ----------
    x : Tensor
        The input data which will be scaled. The shape of x should be
        [*, nfeatures]. Where '*' means any additionaly dimensions.
    mu : float
        The mean of the random noise distribution.
    sigma : float
        The standard deviation of the random noise distribution.
    """
    # Create a Gaussian noise sampler.
    sampler = Normal(mu, sigma)
    # Normally distributed with loc=mu and scale=sigma
    noise = sampler.sample((x.shape[-1],)).to(x.device)
    return x * noise


def jittering(x: Tensor, mu=0.0, sigma=1.0) -> Tensor:
    """
    Add N(0, 1) white noise to the input data.

    Parameters
    ----------
    x : tensor
        The input data which will be jittered. The shape of x should be
        [*, nfeatures]. Where '*' means any additionaly dimensions.
    mu : float
        The mean of the random noise distribution.
    sigma : float
        The standard deviation of the random noise distribution.
    """
    # Create a Gaussian noise sampler.
    sampler = Normal(mu, sigma)
    # Normally distributed with loc=mu and scale=sigma
    noise = sampler.sample((x.shape[-1],)).to(x.device)
    return x + noise


def shuffle(x: Tensor, nsegs=3, seg_min_nfeat=50):
    """
    Shuffle the order of neural features.

    Parameters
    ----------
    x : Tensor
        The input neural data with shape of [nbatches, nsamples, nfeatures].
        If the input x is a 2D tensor with shape [nsamples, nfeatures], this
        function will add a dim on the first dimension before computing, and
        squeeze the adding dimension before return the results.
    nsegs : int
        Number of segments of the original feature dim will be splitted.
        The feature segments will be shuffled to form the new neural data.
    seg_min_nfeat : int
        The minimum number of features of each segment should contain.
    """
    # Check the ndim of the input data.
    if x.ndim == 2:
        unsqueezed = True
        data = x.unsqueeze(0)
    elif x.ndim == 3:
        unsqueezed = False
        data = x
    else:
        raise RuntimeError(f"The ndim of x is {x.ndim}, only 2D or 3D "
                           "input data supported.")
    N = data.shape[-1]

    # Initialize the segment indices
    indices = torch.zeros(nsegs + 1, dtype=torch.int)
    indices[-1] = N

    while 1:
        # Set the limit of the segments start and end.
        # Make sure the first and last feature segment contaion
        # number of features >= seg_min_nfeat.
        low, high = seg_min_nfeat, N - seg_min_nfeat
        # Random generate the indices of each segment.
        indices[1:-1], _ = torch.sort(torch.randint(low, high, (nsegs - 1,)))
        # Check if the length of all segments satisfied the requirement.
        if torch.min(indices[1:] - indices[:-1]) >= seg_min_nfeat:
            break

    _x = torch.zeros_like(data)
    # Random set the order of the segment indices.
    n = 0
    order = torch.randperm(nsegs)
    for i in order:
        ibeg, iend = indices[i], indices[i + 1]
        m = n + (iend - ibeg)

        _x[:, :, n:m] = data[:, :, ibeg:iend]
        # Update the start index of next segment.
        n = m

    # Squeeze back the dimensions.
    if unsqueezed:
        _x = _x.squeeze(0)
    return _x


AUGMENT_FNS = {
    'shuffle':      shuffle,
    'scaling':      scaling,
    'jittering':    jittering,
    'mixup':        mixup
}


# class DataAugment(object):
#     """
#     Data augmentation for the BCI input data.
#     """
#     def permutation(self, x: Tensor, y: Tensor,
#                     nPerm: int = 3, min_seg_length: int = 50):
#         """
#         Permutation the order of the sequence data.

#         Parameters
#         ----------
#         nPerm : int, optional
#             Number of the permutation segements.
#         min_seg_length : int, optional
#             The min segment length when do the permutation.
#         """
#         x_ = torch.zeros(x.shape, device=x.device)
#         y_ = torch.zeros(y.shape, device=y.device)
#         while 1:
#             # SEGS contain the endpoint for each segment.
#             segs = torch.zeros(nPerm + 1, dtype=torch.int)
#             # The first element of SEGS is 0 and the last
#             # element of SEGS is the length of x.
#             segs[1:-1], _ = torch.sort(
#                 torch.randint(min_seg_length, x.shape[0] - min_seg_length,
#                               (nPerm - 1,))
#             )
#             segs[-1] = x.shape[0]
#             # Check if all segments satisfact to
#             # min_seg_length. If not, redo the above step.
#             if (segs[1:] - segs[:-1]).min() > min_seg_length:
#                 break
#         # Permute x and y simutancely.
#         pos = 0
#         indices = torch.randperm(nPerm)
#         for i in indices:
#             temp = x[segs[i]:segs[i + 1]]
#             x_[pos:pos + temp.shape[0]] = temp

#             temp = y[segs[i]:segs[i + 1]]
#             y_[pos:pos + temp.shape[0]] = temp

#             pos += temp.shape[0]
#         return x_, y_

#     def rotation(self, x: Tensor, max_degree=15):
#         """
#         Rotate the Array of the electrodes.
#         """
#         F = self.n_bands
#         # Obtain a random rotate angle.
#         angle = torch.rand(1) * 2 * max_degree - max_degree
#         if angle == 0:
#             return x

#         # The total number of electrodes.
#         N = self.w_array * self.h_array
#         assert N * F == x.size(-1), \
#             'Error occored for the computation of neural '\
#             'array channels.'
#         # Width and height of the array.
#         W = self.w_array
#         H = self.h_array

#         x = x.reshape(-1, N, F).permute(0, 2, 1).reshape(-1, F, H, W)
#         x = TF.rotate(x, angle.item(), TF.InterpolationMode.BILINEAR)
#         x = x.reshape(-1, F, N).permute(0, 2, 1).reshape(-1, N * F)
#         return x

#     def padding(self, x: Tensor, padding: List[int] = None, fill=0,
#                 mode='constant'):
#         """
#         Padding the given neural grid on all sides with the given
#         "PAD" value.

#         Parameters
#         ----------
#         x: Tensor
#             The input neural data. It should be able to reshape to
#             [..., N_BANDs, H_ARRAY, W_ARRAY], where the ... means
#             the leading dimension.
#         padding: int or sequence, optional
#             Padding on each border. If a single int is provided
#             this is used to pad all borders. If sequence of length
#             2 is provided this is the padding on left/right and
#             top/bottom respectively. If a sequence of length 4 is
#             provided this is the padding for the left, top, right
#             and bottom borders respectively. Default: None
#         fill: int, optional
#             The fill vaule for constant fill. This value is only
#             used when MODE='constant'. Default: 0.
#         mode: str, optional
#             Type of padding. Should be one of 'constant', 'edge',
#             'reflect', or 'symmetric'. Default: 'constant'.
#             'constant' - Pads with a constant value, this value is
#                          specified with fill.
#             'edge'     - Pads with the last value at the edge of the
#                          grid. If input a 5D torch Tensor, the last
#                          3 dimensions will be padded instead of the
#                          last 2.
#             'reflect'  - Pads with reflection of grid without repeating
#                          the last value on the edge. For example,
#                          padding [1, 2, 3, 4] with 2 elements on both
#                          sides in reflect mode will result in
#                          [3, 2, 1, 2, 3, 4, 3, 2].
#             'symmetric'- Pads with reflection of image repeating the
#                          last value on the edge. For example, padding
#                          [1, 2, 3, 4] with 2 elements on both sides in
#                          symmetric mode will result in
#                          [2, 1, 1, 2, 3, 4, 4, 3].
#         """
#         F = self.n_bands
#         # The total number of electrodes.
#         N = self.w_array * self.h_array
#         assert N * F == x.size(-1), \
#             'Error occored for the computation of neural '\
#             'array channels.'
#         # Width and height of the array.
#         W, H = self.w_array, self.h_array

#         # Specify the padding if it is none.
#         if padding is None:
#             max_padding = min(W // 2, H // 2) // 3
#             # High value is excluded.
#             padding = torch.randint(1, max_padding + 1, (1,)).item()

#         x = x.reshape(-1, N, F).permute(0, 2, 1).reshape(-1, F, H, W)
#         x = TF.resize(x, [H - padding * 2, W - padding * 2])
#         x = TF.pad(x, padding, fill, mode)
#         x = x.reshape(-1, F, N).permute(0, 2, 1).reshape(-1, N * F)
#         return x

#     def crop(self, x: Tensor, scale: List[float] = (0.08, 1.0),
#              ratio: List[float] = (3. / 4., 4. / 3.)):
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("Scale and ratio should be of kind (min, max).")

#         F = self.n_bands
#         # Width and height of the array.
#         W, H = self.w_array, self.h_array
#         # The total number of electrodes.
#         N = W * H
#         assert N * F == x.size(-1), \
#             'Error occored for the computation of neural '\
#             'array channels.'

#         log_ratio = torch.log(torch.tensor(ratio))
#         i, j = -1, -1
#         for _ in range(10):
#             target_area = N * torch.empty(1).uniform_(scale[0], scale[1])
#             target_area = target_area.item()
#             aspect_ratio = torch.exp(
#                 torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
#             ).item()

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if 0 < w <= W and 0 < h <= H:
#                 i = torch.randint(0, H - h + 1, size=(1,)).item()
#                 j = torch.randint(0, W - w + 1, size=(1,)).item()
#                 break

#         if i == -1 and j == -1:
#             # Fallback to central crop
#             in_ratio = float(W) / float(H)
#             if in_ratio < min(ratio):
#                 w = W
#                 h = int(round(w / min(ratio)))
#             elif in_ratio > max(ratio):
#                 h = H
#                 w = int(round(h * max(ratio)))
#             else:  # whole image
#                 w = W
#                 h = H
#             i = (H - h) // 2
#             j = (W - w) // 2

#         # Random crop the grid according i, j, h, w.
#         x = x.reshape(-1, N, F).permute(0, 2, 1).reshape(-1, F, H, W)
#         x = TF.resized_crop(x, i, j, h, w, (H, W))
#         x = x.reshape(-1, F, N).permute(0, 2, 1).reshape(-1, N * F)
#         return x


# if __name__ == '__main__':
#     torch.manual_seed(42)

#     augmentor = DataAugment(p=1, w_array=8, h_array=9,
#                             n_bands=21, methods=['rotation'])
#     data = torch.randn((1, 267, 1512), device='cuda')
#     rotated = augmentor(data)[0]

#     scio.savemat(
#         'test_rotate.mat',
#         {'data': data.squeeze(0).cpu().numpy(),
#          'rotated': rotated.squeeze(0).cpu().numpy()}
#     )
