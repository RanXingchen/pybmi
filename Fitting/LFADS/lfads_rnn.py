import torch
import torch.nn as nn


class LFADSGRUCell(nn.Module):
    """
    Implements gated recurrent unit used in LFADS generator and controller.
    The parameters of this GRU transforming hidden state are kept separate for
    computing L2 cost (see bullet point 2 of section 1.9 in online methods).
    Also does not create parameters transforming inputs if no inputs exist.
    """
    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super(LFADSGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias

        # Concatenated size
        self._ru_size = self.hidden_size * 2

        # Create parameters for transforming inputs if inputs exist
        if self.input_size > 0:
            # rx ,ux = W(x) (No bias in tensorflow implementation)
            self.fc_x_ru = nn.Linear(self.input_size, self._ru_size, False)
            # cx = W(x) (No bias in tensorflow implementation)
            self.fc_x_c = nn.Linear(self.input_size, self.hidden_size, False)

        # Create parameters transforming hidden state
        # rh, uh = W(h) + b
        self.fc_h_ru = nn.Linear(self.hidden_size, self._ru_size)
        # ch = W(h) + b
        self.fc_rh_c = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, h):
        '''
        Gated Recurrent Unit forward pass with forget bias, weight on inputs
        and hidden state kept separate.

        Parameters
        ----------
        x : Tensor
            GRU input
        h : Tensor
            GRU hidden state

        Returns
        -------
        h_new Tensor
            Updated GRU hidden state
        '''
        # Calculate reset and update gates from input
        if self.input_size > 0 and x is not None:
            r_x, u_x = torch.split(self.fc_x_ru(x), self.hidden_size, dim=1)
        else:
            r_x, u_x = 0, 0
        # Calculate reset and update gates from hidden state
        r_h, u_h = torch.split(self.fc_h_ru(h), self.hidden_size, dim=1)

        # Combine reset and updates gates from hidden state and input
        r = torch.sigmoid(r_x + r_h)
        u = torch.sigmoid(u_x + u_h + self.forget_bias)

        # Calculate candidate hidden state from input
        if self.input_size > 0 and x is not None:
            c_x = self.fc_x_c(x)
        else:
            c_x = 0

        # Calculate candidate hidden state from hadamard product of hidden
        # state and reset gate
        c_rh = self.fc_rh_c(r * h)
        # Combine candidate hidden state vectors
        c = torch.tanh(c_x + c_rh)
        # Return new hidden state as a function of update gate, current hidden
        # state, and candidate hidden state
        return u * h + (1 - u) * c

    def hidden_weight_l2_norm(self):
        norm_ru_w2 = self.fc_h_ru.weight.norm(2).pow(2)
        n_ru = self.fc_h_ru.weight.numel()
        norm_rh_w2 = self.fc_rh_c.weight.norm(2).pow(2)
        n_rh = self.fc_rh_c.weight.numel()
        return norm_ru_w2 / n_ru + norm_rh_w2 / n_rh
