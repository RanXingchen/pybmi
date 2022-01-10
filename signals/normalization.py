import pybmi as bmi
import torch


class Normalization():
    def __init__(self, mode='z-score', dim=-1):
        self.mode = bmi.utils.check_params(
            mode, ['min-max', 'z-score'], 'mode'
        )
        self.dim = dim

    def fit(self, x: torch.Tensor, keepdim=False):
        if self.mode == 'z-score':
            self.mu = x.mean(dim=self.dim, keepdim=keepdim)
            self.sigma = x.std(dim=self.dim, keepdim=keepdim)
        elif self.mode == 'min-max':
            self.min_val, _ = x.min(dim=self.dim, keepdim=keepdim)
            self.max_val, _ = x.max(dim=self.dim, keepdim=keepdim)
        return self

    def apply(self, x: torch.Tensor):
        if self.mode == 'z-score':
            return (x - self.mu) / self.sigma
        elif self.mode == 'min-max':
            return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse(self, x: torch.Tensor):
        if self.mode == 'z-score':
            return x * self.sigma + self.mu
        elif self.mode == 'min-max':
            return x * (self.max_val - self.min_val) + self.min_val
