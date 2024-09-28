import torch
import torch.nn as nn
import torch.nn.functional as F

import data_compression.ops as ops

__all__ = ["GDN"]


class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        num_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = .1,
        reparam_offset: float = 2 ** -18,
    ):
        super().__init__()
        self.inverse = bool(inverse)
        self._beta_min = float(beta_min)
        self._gamma_init = float(gamma_init)
        self._reparam_offset = float(reparam_offset)
        self._pedestal = self._reparam_offset ** 2
        self._beta_bound = (self._beta_min + self._reparam_offset ** 2) ** 0.5
        self._gamma_bound = self._reparam_offset
        self._gamma = nn.Parameter(torch.FloatTensor(num_channels, num_channels, 1, 1))
        self._beta = nn.Parameter(torch.FloatTensor(num_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nc = self._gamma.shape[0]
        self._beta.data = torch.sqrt(torch.ones(nc) + self._pedestal)
        self._gamma.data = torch.sqrt(self._gamma_init * torch.eye(nc).view(nc, nc, 1, 1) + self._pedestal)

    @property
    def gamma(self):
        return torch.pow(ops.lower_bound(self._gamma, self._gamma_bound), 2) - self._pedestal

    @property
    def beta(self):
        return torch.pow(ops.lower_bound(self._beta, self._beta_bound), 2) - self._pedestal

    def forward(self, x):
        norm = F.conv2d(x ** 2, self.gamma, self.beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        x = x * norm
        return x