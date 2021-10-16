from torch import Tensor


def global_variance_equalization(reference: Tensor, estimated: Tensor,
                                 mean: bool = False):
    """
    Computing global equalization factor to alleviate the over-smooth
    problem.

    Parameters
    ----------
    reference: Tensor
        The reference mel cepstrum or spectrograms.
        Shape: [length, features]
    estimated: Tensor
        The estimated mel cepstrum or spectrograms, which had the same
        shape with REFERENCE.
    mean: bool, optional
        Computing mean GV over all features or not. Default: False.
    """
    assert reference.shape == estimated.shape
    
    if mean:
        gv_reference = ((reference - reference.mean()) ** 2).mean()
        gv_estimated = ((estimated - estimated.mean()) ** 2).mean()
    else:
        gv_reference = ((reference - reference.mean(dim=0)) ** 2).mean(dim=0)
        gv_estimated = ((estimated - estimated.mean(dim=0)) ** 2).mean(dim=0)
    gv_factor = (gv_reference / gv_estimated).sqrt()
    return gv_factor
