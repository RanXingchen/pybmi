import torch.nn as nn

from torch import Tensor
from pybmi.utils import check_params


class _Loss(nn.Module):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction


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
            return kl_loss.sum() / m_p.shape[0]
