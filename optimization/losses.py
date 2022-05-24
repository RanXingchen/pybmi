import torch.nn as nn
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.loss import _Loss
from pybmi.utils import check_params
from ..signals.metrics import ssim


class GaussianKLDivLoss(nn.Module):
    """
    log p(x|z) + KL(q||p) terms for Gaussian posterior and Gaussian prior. See
    eqn 10 and Appendix B in VAE for latter term,
    http://arxiv.org/abs/1312.6114

    The log p(x|z) term is the reconstruction error under the model.
    The KL term represents the penalty for passing information from the encoder
    to the decoder.
    To sample KL(q||p), we simply sample ln(q) - ln(p) by drawing samples
    from q and averaging.
    """
    def __init__(self, reduction: str = 'mean'):
        r"""
        Create a lower bound in three parts, normalized reconstruction cost,
        normalized KL divergence cost, and their sum.

        E_q[ln p(z_i | z_{i+1}) / q(z_i | x)
            \int q(z) ln p(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_p^2) + \
            sigma_q^2 / sigma_p^2 + (mean_p - mean_q)^2 / sigma_p^2)

            \int q(z) ln q(z) dz = - 0.5 ln(2pi) - 0.5 \sum (ln(sigma_q^2) + 1)

        Parameters
        ----------
        reduction : string, optional
            Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'.

                'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                        elements in the output.
                'sum' : the output will be summed.
        """
        super(GaussianKLDivLoss, self).__init__()
        self.reduction = check_params(reduction, ['none', 'mean', 'sum'],
                                      'reduction')

    def forward(self, m_p: Tensor, s_p: Tensor, m_q: Tensor,
                s_q: Tensor) -> Tensor:
        """
        Parameters
        ----------
        m_p : Tensor
            Mean for the prior.
        s_p : Tensor
            Log-variance for the prior.
        m_q : Tensor
            Mean for the posterior.
        s_q : Tensor
            Log-variance for the posterior.
        """
        kl_loss = 0.5 * (s_p - s_q + (s_q - s_p).exp() +
                         ((m_q - m_p) / (0.5 * s_p).exp()).pow(2) - 1.0)
        if self.reduction == 'none':
            return kl_loss
        elif self.reduction == 'sum':
            return kl_loss.sum()
        else:
            return kl_loss.mean(dim=0).sum()


class FocalLoss(nn.Module):
    """
    Focal Loss: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: Tensor, gamma=2, reduction='mean'):
        """
        Parameters
        ----------
        alpha : Tensor, optional
            The weight of classes. Each represent the the weight of one class.
        gamma : float, optional
            The attenuation factor of the probability. Used to balance the
            weight of difficult and easy samples. Gamma is more important
            than alpha, the larger of gamma, the smaller of alpha.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x: Tensor, labels: Tensor):
        """
        Usage is same as nn.CrossEntropyLoss:
        >>> criteria = FocalLoss()
        >>> logits = torch.randn(8, 19, 384, 384)
        >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
        >>> loss = criteria(logits, lbs)
        """
        ce_loss = F.cross_entropy(x, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha.gather(0, labels) * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()


class LeCamLoss(nn.Module):
    def __init__(self, decay=0.9, start_iter=1000):
        """
        Apply Lecam regularization for GAN's discriminator.

        The method is discribed at the paper "Regularizing Generative
        Adversarial Networks under Limited Data":
        https://arxiv.org/pdf/2104.03310.pdf

        This code is reference from https://github.com/google/lecam-gan.

        Parameters
        ----------
        decay : float
            Decay for the exponential moving average.
        start_iter : int
            Start iteration to apply lecam regularization.
        """
        super(LeCamLoss, self).__init__()
        self.running_Dr = 0.0
        self.running_Df = 0.0

        self.decay = decay
        self.start_iter = start_iter

    def update(self, dr: float, df: float, iter):
        """
        Update the state of the exponential moving average.

        Parameters
        ----------
        dr : float
            The average value of discriminator output for the real data
            of current iter.
        df : float
            The average value of discriminator output for the fake data
            of current iter.
        """
        decay = 0.0 if iter < self.start_iter else self.decay

        self.running_Dr = decay * self.running_Dr + (1 - decay) * dr
        self.running_Df = decay * self.running_Df + (1 - decay) * df

    def forward(self, D_real: Tensor, D_fake: Tensor):
        loss = torch.mean(F.relu(D_real - self.running_Df).pow(2)) + \
            torch.mean(F.relu(self.running_Dr - D_fake).pow(2))
        return loss


class SSIMLoss(_Loss):
    """
    Creates a criterion that measures the structural similarity index error
    between each element in the input x and target y.

    To match performance with skimage and tensorflow set 'downsample' = True.

    This SSIM Loss is a copy from:
    https://github.com/photosynthesis-team/piq

    Parameters
    ----------
    kernel_size : int, optional
        By default, the mean and covariance of a pixel is obtained by
        convolution with given filter_size. Default: 11.
    kernel_sigma : float, optional
        Standard deviation for Gaussian kernel. Default: 1.5.
    k1 : float, optional
        Coefficient related to c1 in the above equation. Default: 0.01.
    k2 : float, optional
        Coefficient related to c2 in the above equation. Default: 0.03.
    downsample : bool, optional
        Perform average pool before SSIM computation. Default: True.
    reduction : str, optional
        Specifies the reduction type:
        ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
    max_value : float, optional
        Maximum value range of images (usually 1.0 or 255). Default: 1.0.

    Examples
    --------
    >>> loss = SSIMLoss()
    >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
    >>> y = torch.rand(3, 3, 256, 256)
    >>> output = loss(x, y)
    >>> output.backward()

    References
    ----------
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: From error visibility to structural similarity.
    IEEE Transactions on Image Processing, 13, 600-612.
    https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
    DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self,
                 kernel_size: int = 11,
                 kernel_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 downsample: bool = True,
                 reduction: str = 'mean',
                 max_value: float = 1.0) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked
        # within the ssim function anyway. However, this check allows to
        # fail fast when the loss is being initialised and training has not
        # been started.
        assert kernel_size % 2 == 1, \
            f'Kernel size must be odd, got [{kernel_size}]'

        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.max_value = max_value

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computation of Structural Similarity (SSIM) index as a loss function.

        Parameters
        ----------
        x : Tensor
            An input tensor. Shape :math:`(N, C, H, W)`.
        y : Tensor
            A target tensor. Shape :math:`(N, C, H, W)`.

        Returns
        -------
        Value of SSIM loss to be minimized, i.e ``1 - ssim`` in [0, 1] range.
        """
        score = ssim(x, y,
                     kernel_size=self.kernel_size,
                     kernel_sigma=self.kernel_sigma,
                     downsample=self.downsample,
                     max_value=self.max_value,
                     reduction=self.reduction,
                     full=False,
                     k1=self.k1,
                     k2=self.k2)
        return torch.ones_like(score) - score
