from distutils.log import error
from typing import List
from torch import Tensor
from ...utils import check_params

import torch
import torch.nn.functional as F


def ssim(x: Tensor,
         y: Tensor,
         kernel_size: int = 11,
         kernel_sigma: float = 1.5,
         max_value: float = 1.0,
         reduction: str = 'mean',
         full: bool = False,
         downsample: bool = True,
         k1: float = 0.01,
         k2: float = 0.03) -> List[Tensor]:
    """
    Interface of Structural Similarity (SSIM) index.

    Inputs supposed to be in range [0, max_value].
    To match performance with skimage and tensorflow set 'downsample' = True.

    This ssim function is a copy from:
    https://github.com/photosynthesis-team/piq

    Parameters
    ----------
    x : Tensor
        An input tensor. Shape: (N, C, H, W) or (C, H, W) or (H, W).
    y : Tensor
        A target tensor. Shape: (N, C, H, W) or (C, H, W) or (H, W).
    kernel_size : int, optional
        The side-length of the sliding window used in comparison.
        Must be an odd value. Default: 11.
    kernel_sigma : float, optional
        Sigma of normal distribution. Default: 1.5.
    max_value : float, optional
        Maximum value range of images (usually 1.0 or 255). Default: 1.0.
    reduction : str, optional
        Specifies the reduction type: 'none' | 'mean' | 'sum'.
        Default:'mean'.
    full : bool, optional
        Return cs map or not. Default: False.
    downsample : bool, optional
        Perform average pool before SSIM computation. Default: True
    k1 : float, optional
        Algorithm parameter, K1 (small constant). Defalut: 0.01.
    k2 : float, optional
        Algorithm parameter, K2 (small constant). Try a larger K2 constant
        (e.g. 0.4) if you get a negative or NaN results. Default: 0.03.

    Returns
    -------
    Value of Structural Similarity (SSIM) index. In case of 5D input tensors,
    complex value is returned as a tensor of size 2.

    References
    ----------
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: From error visibility to structural similarity.
    IEEE Transactions on Image Processing, 13, 600-612.
    https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
    DOI: `10.1109/TIP.2003.819861`
    """
    assert kernel_size % 2 == 1, \
        f'Kernel size must be odd, got [{kernel_size}]'
    # The shape of x and y should be same.
    assert x.ndim == y.ndim, "The shape of x and y should be equal!"

    # Check the type of reduction.
    reduction = check_params(reduction, ['none', 'mean', 'sum'], 'reduction')

    x = x / max_value
    y = y / max_value

    # Check the number of dim of x and y.
    dim_B_squeezed = False
    if x.ndim == 2:
        # Lack of batch dim and channel dim.
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        dim_B_squeezed = True
    elif x.ndim == 3:
        # !WARNING: The shape of x can't be [N, H, W]...
        # Lack of batch dim.
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        dim_B_squeezed = True
    elif x.ndim > 4:
        error(f'Do not support number of dimension {x.ndim}')

    # Average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = _gaussian_filter(kernel_size, kernel_sigma).\
        repeat(x.size(1), 1, 1, 1).to(y)
    ssim_map, cs_map = _ssim_per_ch(x, y, kernel, k1=k1, k2=k2)
    # Average the channel dim.
    ssim_val, cs = ssim_map.mean(1), cs_map.mean(1)

    if reduction == 'none' and dim_B_squeezed:
        ssim_val, cs = ssim_val.squeeze(0), cs.squeeze(0)
    elif reduction == 'mean':
        ssim_val, cs = ssim_val.mean(dim=0), cs.mean(dim=0)
    elif reduction == 'sum':
        ssim_val, cs = ssim_val.sum(dim=0), cs.sum(dim=0)

    if full:
        return [ssim_val, cs]

    return ssim_val


def _gaussian_filter(kernel_size: int, sigma: float,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Returns 2D Gaussian kernel N(0,`sigma`^2)

    Parameters
    ----------
    size : int
        Size of the kernel.
    sigma : float
        Std of the distribution.
    dtype : torch.dtype, optional
        Type of tensor to return. Default: torch.float32
    Returns
    -------
    gaussian_kernel : Tensor
        The Gaussian kernel with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def _ssim_per_ch(x: Tensor, y: Tensor, k: Tensor, k1: float = 0.01,
                 k2: float = 0.03):
    """
    Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Parameters
    ----------
    x : Tensor
        An input tensor. Shape :math:`(N, C, H, W)`.
    y : Tensor
        A target tensor. Shape :math:`(N, C, H, W)`.
    k : Tensor
        2D Gaussian kernel.
    k1 : float, optional
        Algorithm parameter, K1 (small constant, see [1]). Default: 0.01.
    k2 : float, optional
        Algorithm parameter, K2 (small constant, see [1]). Try a larger K2
        constant (e.g. 0.4) if you get a negative or NaN results.
        Default: 0.03.

    Returns
    -------
    Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < k.size(-1) or x.size(-2) < k.size(-2):
        raise ValueError('Kernel size can\'t be greater than input size. '
                         f'Input size: {x.size()}. '
                         f'Kernel size: {k.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    nch = x.size(1)
    mu_x = F.conv2d(x, weight=k, stride=1, padding=0, groups=nch)
    mu_y = F.conv2d(y, weight=k, stride=1, padding=0, groups=nch)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, k, stride=1, padding=0, groups=nch) - mu_xx
    sigma_yy = F.conv2d(y ** 2, k, stride=1, padding=0, groups=nch) - mu_yy
    sigma_xy = F.conv2d(x * y, k, stride=1, padding=0, groups=nch) - mu_xy

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs
