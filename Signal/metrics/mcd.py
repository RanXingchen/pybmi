import torch
import math


ALPHA = 10 * math.sqrt(2) / math.log(10)

def MCD(x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor = None):
    """
    Computing mel-cepstral distortion for two same length sequences.

    Parameters
    ----------
    x : Tensor
        The true mel-cepstral.
    y : Tensor
        The predicted mel-cepstral.
    mask : BoolTensor, optional.
        A bool tensor that indicate the silence regions of the sequence.
        In which True index will be remained, and False will be masked out.
        Default: None.
    """
    assert x.size(0) == y.size(0), "The length of two MELs not equal."
    # ignore 0th cepstral component
    _x, _y = x[:, 1:], y[:, 1:]

    diff = _x - _y
    mcd_seq = torch.inner(diff, diff).diag().sqrt()

    # Average for each frame.
    if mask is not None:
        mcd = ALPHA * mcd_seq.masked_select(mask).sum() / mask.int().sum()
    else:
        mcd = ALPHA * mcd_seq.sum() / mcd_seq.size(0)
    return mcd


if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn(132, 25)
    b = torch.randn(132, 25)
    mask = torch.zeros((132,), dtype=bool)
    mask[:10], mask[-10:] = True, True
    mcd = MCD(a, b, mask)
    print(f'The Mel-Cepstral Distortion is {mcd}.')
