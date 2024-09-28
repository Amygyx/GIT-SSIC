import abc, math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_compression.ops as ops
from data_compression.distributions import helpers
try:
    from data_compression._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from data_compression import rans
except:
    pass
import numpy as np

def pmf_to_quantized_cdf(pmf, precision: int = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class ContinuousEntropyModelBase(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                 prior=None,
                 tail_mass=2**-8,
                 range_coder_precision=16):
        super().__init__()
        self._prior = prior
        self._tail_mass = float(tail_mass)
        self._range_coder_precision = int(range_coder_precision)
        try:
            self._encoder = rans.RansEncoder()
            self._decoder = rans.RansDecoder()
        except:
            pass
        # shape = self.prior.batch_shape
        # # print(shape)
        self.register_buffer("_cdf_shape", torch.IntTensor(2).zero_())
        # self.register_buffer("_cdf", torch.IntTensor(*shape))
        # self.register_buffer("_cdf_offset", torch.IntTensor(*shape))
        # self.register_buffer("_cdf_length", torch.IntTensor(*shape))

        self._decompress_time1 = 0
        self._decompress_time2 = 0
        self._decompress_time3 = 0
        self._decompress_time4 = 0

    @property
    def prior(self):
        """Prior distribution, used for deriving range coding tables."""
        if self._prior is None:
            raise RuntimeError(
            "This entropy model doesn't hold a reference to its prior "
            "distribution.")
        return self._prior

    @property
    def cdf(self):
        return self._cdf

    @property
    def cdf_offset(self):
        return self._cdf_offset

    @property
    def cdf_length(self):
        return self._cdf_length

    @property
    def tail_mass(self):
        return self._tail_mass

    @property
    def range_coder_precision(self):
        return self._range_coder_precision

    def _log_prob_from_prior(self, prior, indexes):
        return prior.log_prob(indexes)

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)

    def compress(self, indexes, cdf_indexes):
        string = self.encode_with_indexes(
            indexes.reshape(-1).int().tolist(),
            cdf_indexes.reshape(-1).int().tolist(),
            self.cdf.tolist(),
            self.cdf_length.reshape(-1).int().tolist(),
            self.cdf_offset.reshape(-1).int().tolist(),
        )

        # self.cdf_indexes = cdf_indexes
        return string

    def to_list(self):
        self.cdf_list = self.cdf.tolist()
        self.cdf_length_list = self.cdf_length.reshape(-1).int().tolist()
        self.cdf_offset_list = self.cdf_offset.reshape(-1).int().tolist()

    def decompress(self, string, cdf_indexes, cdf_shape=None):
        t0 = time.time()
        if hasattr(self, "cdf_list"):
            cdf = self.cdf_list
            cdf_length = self.cdf_length_list
            cdf_offset = self.cdf_offset_list
        else:
            cdf = self.cdf.tolist()
            cdf_length = self.cdf_length.reshape(-1).int().tolist()
            cdf_offset = self.cdf_offset.reshape(-1).int().tolist()
        t1 = time.time()
        if isinstance(cdf_indexes, list):
            values = self.decode_with_indexes(
                string,
                cdf_indexes,
                cdf,
                cdf_length,
                cdf_offset,
            )
            values = torch.tensor(
                values, device='cuda', dtype=torch.float32
            ).reshape(cdf_shape)
        else:
            values = self.decode_with_indexes(
                string,
                cdf_indexes.reshape(-1).int().tolist(),
                cdf,
                cdf_length,
                cdf_offset,
            )
            values = torch.tensor(
                values, device=cdf_indexes.device, dtype=torch.float32
            ).reshape(cdf_indexes.shape)
        t2 = time.time()

        self._decompress_time1 += t1 - t0
        self._decompress_time2 += t2 - t1
        # self._decompress_time3 += t3 - t2
        # self._decompress_time4 += t4 - t3

        # assert cdf_indexes.cuda().equal(self.cdf_indexes.cuda())
        return values

    def _fix_tables(self):
        cdf, cdf_offset, cdf_length = self._build_tables(
            self.prior, self.range_coder_precision)

        self._cdf_shape.data = torch.IntTensor(list(cdf.shape)).to(cdf.device)
        self._init_tables()
        self._cdf.data = cdf.int()
        self._cdf_offset.data = cdf_offset.int()
        self._cdf_length.data = cdf_length.int()

    def _init_tables(self): # TODO: check cdf device
        shape = self._cdf_shape.tolist()
        device = self._cdf_shape.device
        self.register_buffer("_cdf", torch.IntTensor(*shape).to(device))
        self.register_buffer("_cdf_offset", torch.IntTensor(*shape[:1]).to(device))
        self.register_buffer("_cdf_length", torch.IntTensor(*shape[:1]).to(device))

    def _build_tables(self, prior, precision):
        """Computes integer-valued probability tables used by the range coder.
        These tables must not be re-generated independently on the sending and
        receiving side, since small numerical discrepancies between both sides can
        occur in this process. If the tables differ slightly, this in turn would
        very likely cause catastrophic error propagation during range decoding.
        Args:
          prior: distribution
        Returns:
          CDF table, CDF offsets, CDF lengths.
        """
        lower_tail = prior.lower_tail(self.tail_mass)
        upper_tail = prior.upper_tail(self.tail_mass)

        minima = torch.floor(lower_tail).int()
        maxima = torch.ceil(upper_tail).int()
        # PMF starting positions and lengths.
        pmf_start = minima
        pmf_length = maxima - minima + 1
        device = pmf_start.device

        # Sample the densities in the computed ranges, possibly computing more
        # samples than necessary at the upper end.
        max_length = pmf_length.max().int().item()
        samples = torch.arange(max_length, device=device)
        samples = samples.view([-1] + len(pmf_start.shape) * [1])
        samples = samples + pmf_start.unsqueeze(0)

        pmf = prior.prob(samples)
        num_pmfs = pmf_start.numel()

        # Collapse batch dimensions of distribution.
        pmf = torch.reshape(pmf, [max_length, num_pmfs])
        pmf = pmf.transpose(0, 1)

        pmf_length = pmf_length.view([num_pmfs])
        cdf_length = pmf_length + 2
        cdf_offset = minima.view([num_pmfs])

        cdf = torch.zeros(
            [num_pmfs, max_length + 2], dtype=torch.int32, device=device
        )
        for i, p in enumerate(pmf):
            p = p[:pmf_length[i]]
            overflow = (1. - p.sum(dim=0, keepdim=True)).clamp_min(0)
            p = torch.cat([p, overflow], dim=0)
            c = pmf_to_quantized_cdf(p, precision)
            cdf[i, :c.shape[0]] = c
        return cdf, cdf_offset, cdf_length


class ContinuousUnconditionalEntropyModel(ContinuousEntropyModelBase):

    def __init__(self,
                 prior,
                 tail_mass=2**-8,
                 range_coder_precision=16):
        super().__init__(
            prior=prior,
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision,
        )

    def forward(self, indexes, draw=False, keep_batch=False):
        log_probs = self._log_prob_from_prior(self.prior, indexes)
        if keep_batch:
            bits = torch.sum(log_probs, dim=(1, 2, 3)) / (-math.log(2))
        else:
            bits = torch.sum(log_probs) / (-math.log(2))
        if draw:
            return bits, log_probs
        return bits 

    @staticmethod
    def _build_cdf_indexes(shape):
        dims = len(shape)
        B = shape[0]
        C = shape[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(B, 1, *shape[2:])

    def compress(self, indexes):
        cdf_indexes = self._build_cdf_indexes(indexes.shape)
        return super().compress(indexes, cdf_indexes)

    def decompress(self, string, shape):
        shape = [1, self.cdf.shape[0]] + shape
        cdf_indexes = self._build_cdf_indexes(shape).to(self.cdf.device)
        return super().decompress(string, cdf_indexes)


class ContinuousConditionalEntropyModel(ContinuousEntropyModelBase):

    def __init__(self,
                 prior_fn,
                 param_tables,
                 tail_mass=2**-8,
                 range_coder_precision=16):
        self._prior_fn = prior_fn
        self._param_tables = param_tables
        super().__init__(
            prior=self._make_range_coding_prior(self.param_tables),
            tail_mass=tail_mass,
            range_coder_precision=range_coder_precision,
        )

    @property
    def param_tables(self):
        return self._param_tables

    @property
    def prior_fn(self):
        return self._prior_fn

    def _make_prior(self, params):
        return self.prior_fn(**params)

    def _make_range_coding_prior(self, param_tables):
        try:
            params = [torch.FloatTensor(v) for v in param_tables.values()]
            params = torch.meshgrid(*params, indexing="ij")
            params = {
                k: v.view(-1) for k, v in zip(param_tables.keys(), params)
            }
            return self._make_prior(params)
        except:
            pass


    def _normalize_params(self, params):
        for k, v in self.param_tables.items():
            params[k] = ops.lower_bound(params[k], v[0])
            params[k] = ops.upper_bound(params[k], v[-1])
        return params

    def forward(self, indexes, draw=False, keep_batch=False, **params):
        assert sorted(params.keys()) == sorted(self.param_tables.keys())
        params = self._normalize_params(params)
        prior = self._make_prior(params)

        log_probs = self._log_prob_from_prior(prior, indexes)
        if keep_batch:
            bits = torch.sum(log_probs, dim=(1, 2, 3)) / (-math.log(2))
        else:
            bits = torch.sum(log_probs) / (-math.log(2))
        if draw:
            return bits, log_probs
        return bits

    def _build_cdf_indexes(self, params):
        indexes_range = []
        indexes = []
        for k, v in params.items():
            table = torch.Tensor(self.param_tables[k]).to(v.device)
            dist = (v.view(-1, 1) - table.view(1, -1)).abs()
            indexes.append(dist.min(dim=-1)[1].view(v.shape))
            indexes_range.append(len(table))
        ## TODO: multi indexes stride
        assert len(indexes) == 2
        indexes = indexes[0] * indexes_range[0] + indexes[1]
        return indexes

    def compress(self, indexes, **params):
        params = self._normalize_params(params)
        cdf_indexes = self._build_cdf_indexes(params)
        return super().compress(indexes, cdf_indexes)

    def pre_decompress(self, **params):
        params = self._normalize_params(params)
        cdf_indexes = self._build_cdf_indexes(params)
        return cdf_indexes
    def decompress(self, string, **params):
        if 'scale' in params.keys():
            t0 = time.time()
            params = self._normalize_params(params)
            cdf_indexes = self._build_cdf_indexes(params)
            t1 = time.time()
            self._decompress_time3 += t1 - t0
            cdf_shape = None
        else:
            cdf_indexes = params['cdf_indexes']
            cdf_shape = params['cdf_shape']
        out = super().decompress(string, cdf_indexes, cdf_shape)
        return out