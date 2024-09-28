import math
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal as _Normal
from torch.distributions.laplace import Laplace as _Laplace
from torch.distributions.categorical import Categorical as _Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

import data_compression.ops as ops
from torch.distributions.distribution import Distribution
from data_compression.distributions.distribution import DeepDistribution
from data_compression.distributions import special_math


class Normal(_Normal):
    def enumerate_support(self, expand=True):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _z(self, x):
        """Standardize input `x` to a unit normal."""
        x = (x - self.loc) / self.scale
        return x

    def log_cdf(self, x):
        return special_math.log_ndtr(self._z(x))

    def log_survival_function(self, x):
        return special_math.log_ndtr(-self._z(x))

    def lower_tail(self, tail_mass):
        return self.icdf(torch.tensor(tail_mass/2))

    def upper_tail(self, tail_mass):
        return self.icdf(torch.tensor(1-tail_mass/2))


class DeepNormal(DeepDistribution):
    def __init__(self, batch_shape):
        super().__init__()
        self._make_parameters(batch_shape)
        self._base = Normal(
            loc=self.loc,
            scale=self.scale
        )

    def _make_parameters(self, batch_shape):
        self.loc = nn.Parameter(torch.zeros(*batch_shape))
        self.scale = nn.Parameter(torch.ones(*batch_shape))


class DeepMixtureNormal(DeepDistribution):
    def __init__(self, batch_shape, num_mixture=3):
        super().__init__()
        self.num_mixture = num_mixture
        self._make_parameters()
        self._base = MixtureSameFamily(
            mixture_distribution=_Categorical(logits=self.weight),
            component_distribution=_Normal(loc=self.loc, scale=self.scale)
        )

    def _make_parameters(self):
        shape = self.batch_shape + [self.num_mixture]
        self.loc = nn.Parameter(torch.zeros(*shape)._uniform(-1, 1))
        self.scale = nn.Parameter(torch.ones(*shape))
        self.weight = nn.Parameter(torch.ones(*shape))
