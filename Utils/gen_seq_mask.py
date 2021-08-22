import torch
from typing import Optional


def generate_sequence_padding_mask(seq_lens: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Generate the sequence padding mask by sequence lengths.

    Parameters
    ----------
    seq_lens : Tensor or ndarray
        This parameter contain the true length of each sample in current batch.
        Note that SEQ_LENS has to be sorted, which means that the first element
        is the maximum sequence length. LEN(SEQ_LENS) is the batch size.

    Returns
    -------
    mask : Tensor
        The True of the mask means masked and False means not masked.
    """
    N, S = seq_lens.shape[0], seq_lens[0]
    mask = torch.arange(0, S).type_as(seq_lens).unsqueeze(0).expand(N, S)
    mask = torch.lt(mask, seq_lens.unsqueeze(1))
    return ~mask