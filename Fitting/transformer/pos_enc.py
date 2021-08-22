import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    r"""
    Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    Parameters
    ----------
    d_model : int
        The embed dim. Note that the embed dim should be divideable by 2.
    dropout : float, optional
        The dropout value (default=0.1).
    max_len : int, optional
        The max length of the incoming sequence (default=5000).

    Examples
    --------
    >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add the batch dim.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Inputs of forward function.

        Parameters
        ----------
        x : Tensor
            The sequence fed to the positional encoder model.
            The shape of x is [batch size, sequence length, embed dim]

        Returns
        -------
        output: Tensor
            The input data that added the positional information.
            Shape: [batch size, sequence length, embed dim]

        Examples
        --------
        >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)