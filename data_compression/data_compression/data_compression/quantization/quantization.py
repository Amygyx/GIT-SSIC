import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal as _Normal

import data_compression.ops as ops

def l2_dist(x, code_book):
    ## x: B, K
    ## code_book: 1, N, K or B, N, K
    ## dist: B, N
    B, K = x.shape
    N = code_book.shape[1]

    x = x.unsqueeze(-1)
    if code_book.shape[0] == 1:
        factor = B * K * N // (128 * 768 * 10000)
        x = x.chunk(factor + 1, dim=0) if factor >= 1 else [x]
        dist = []
        for xi in x:
            dist.append(xi.pow(2).sum(dim=1, keepdim=True) \
                        + code_book.pow(2).sum(dim=2, keepdim=True) \
                        - 2 * code_book.matmul(xi))
        dist = torch.cat(dist, dim=0)
    else:
        # dist = x.pow(2).sum(dim=1, keepdim=True) \
        #        + code_book.pow(2).sum(dim=2, keepdim=True) \
        #        - 2 * torch.einsum('bnk,bkj->bnj', [code_book, x])
        dist = x.pow(2).sum(dim=1, keepdim=True) \
               + code_book.pow(2).sum(dim=2, keepdim=True) \
               - 2 * code_book.matmul(x)
    dist = dist.squeeze(-1)
    return dist


class UniformNoise(nn.Module):
    def __init__(self, step=1.):
        super().__init__()
        self.step = step

    def quantize(self, x):
        return torch.round(x/self.step) * self.step

    def dequanztize(self, x):
        return x

    def forward(self, x, training):
        if training:
            half = self.step/2
            noise = x.new(x.shape).uniform_(-half, half)
            indexes = x + noise
            x_hat = indexes
            return x_hat, indexes
        else:
            indexes = self.quantize(x)
            x_hat = self.dequanztize(indexes)
            return x_hat, indexes


class UniformQuantization(nn.Module):
    def __init__(self, step=1.):
        super().__init__()
        self.step = step

    def quantize(self, x, offset=None):
        if offset is not None:
            x = x - offset
        return torch.round(x / self.step)

    def dequantize(self, indexes, offset=None):
        if offset is not None:
            indexes = indexes + offset
        return indexes * self.step

    def forward(self, x, offset=None, noisy=False):
        if offset is not None:
            x = x - offset

        if noisy:
            half = self.step/2
            quant_noise = x.new(x.shape).uniform_(-half, half)
            x_hat = x + quant_noise
        else:
            x_hat = torch.round(x / self.step) * self.step
            if x.requires_grad:
                x_hat = x + (x_hat - x).detach()

        indexes = x_hat
        if offset is not None:
            x_hat = x_hat + offset

        return x_hat, indexes


from scipy.special import gamma
def sphere_volume(r, n):
    coeff = torch.pi ** (n/2.) / gamma(n/2. + 1)
    return coeff * r ** n

def sphere_radius(v, n):
    coeff = gamma(n/2. + 1) / torch.pi ** (n/2.)
    return (coeff * v) ** (1./n)


class FullSearchVectorQuantization(nn.Module):
    def __init__(self, ncb, cb_size, ndim):
        super().__init__()
        code_book = torch.zeros(ncb, cb_size, ndim)
        code_book = code_book.uniform_(-1/2, 1/2)
        self.code_book = nn.Parameter(code_book)

    def init_with_samples(self, samples):
        ## samples: ncb, cb_size, ndim
        assert samples.shape == self.code_book.shape
        self.code_book.data = samples

    def l2_dist(self, x, code_book):
        ## x: ncb, npoint, ndim
        ## code_book: ncb, cb_size, ndim
        ## dist: ncb, npoint, cb_size

        dist = x.pow(2).sum(dim=2).unsqueeze(-1) \
               + code_book.pow(2).sum(dim=2).unsqueeze(-2)
        code_book = code_book.permute(0, 2, 1)
        # dist = dist - 2*torch.bmm(x, code_book)
        dist = dist - 2 * torch.einsum('abc,acd->abd', [x, code_book])
        return dist

    def find_nearest(self, x, rate_bias=None):
        ncb, npoint, ndim = x.shape
        code_book = self.code_book  ## ncb, cb_size, ndim

        dist = self.l2_dist(x, code_book)  ## ncb, npoint, cb_size
        if rate_bias is None:
            dist = dist / ndim
        else:
            dist = (rate_bias + dist) / ndim

        index = dist.argmin(dim=-1, keepdim=True).detach().long()  ## ncb, npoint, 1
        one_hot = torch.zeros_like(dist)
        one_hot = one_hot.scatter_(-1, index, 1.0)  ## ncb, npoint, cb_size

        x_hat = torch.einsum('abd,adc->abc', one_hot, code_book)
        return x_hat, one_hot, dist

    def forward(self, x, rate_bias=None):
        x_hat, one_hot, dist = self.find_nearest(x, rate_bias)
        return x_hat, one_hot, dist




class LatticeVectorQuantization(nn.Module):
    def __init__(self, N, lattice='optimal', noise_type='original'):
        super().__init__()
        self.noise_type = noise_type
        self.N = N
        if lattice=='optimal':
            if N == 1:
                basis = torch.tensor([[2]])
            elif N == 2:
                basis = torch.tensor([[2, 0],
                                      [1, 3 ** 0.5]])
            elif N == 3:
                basis = torch.tensor([[2, 0, 0],
                                      [-1, 1, 0],
                                      [0, -1, 1]])
            elif N == 4:
                # basis = torch.tensor([[2, 0, 0, 0],
                #                       [1, -1, 0, 0],
                #                       [0, 1, -1, 0],
                #                       [0, 0, 1, -1]]) ## D4
                basis = torch.tensor([[.141421356237E+01, 0, 0, 0],
                                      [-.707106781187E+00, .122474487139E+01, 0, 0],
                                      [0, -.816496580928E+00, .115470053838E+01, 0],
                                      [0, 0, -.866025403784E+00, .111803398875E+01]])  ## A4
                # basis = torch.tensor([[2, 0, 0, 0],
                #                       [0, 2, 0, 0],
                #                       [0, 0, 2, 0],
                #                       [0, 0, 0, 2]]) ## Z4
            else:
                assert N == 8
                # basis = torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0],
                #                       [-1, 1, 0, 0, 0, 0, 0, 0],
                #                       [0, -1, 1, 0, 0, 0, 0, 0],
                #                       [0, 0, -1, 1, 0, 0, 0, 0],
                #                       [0, 0, 0, -1, 1, 0, 0, 0],
                #                       [0, 0, 0, 0, -1, 1, 0, 0],
                #                       [0, 0, 0, 0, 0, -1, 1, 0],
                #                       [.5, .5, .5, .5, .5, .5, .5, .5]]) ## E8
                basis = torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 2, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 2, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 2, 0, 0, 0, 0],
                                      [1, 1, 1, 0, 1, 0, 0, 0],
                                      [1, 0, 1, 1, 0, 1, 0, 0],
                                      [1, 0, 0, 1, 1, 0, 1, 0],
                                      [1, 0, 0, 0, 1, 1, 0, 1]]) # E8 code theory version
        else:
            assert lattice == 'eye'
            basis = torch.eye(N)
        self._basis = nn.Parameter(basis.float()/2)  ## N, K
        self.grid = self.build_grid()


    @property
    def basis(self):
        basis = self._basis
        basis = basis.tril()
        basis = basis / basis.det().abs()
        # basis = basis / basis.pow(2).sum(1).view(-1, 1).sqrt()
        # return basis.detach()
        return basis

    @property
    def voronoi_volume(self):
        return self.basis.det().abs()


    @property
    def index_step(self):
        basis = self.basis
        return basis.diagonal()

    def build_grid(self):
        x = torch.arange(2)
        x = [x] * self._basis.shape[0]
        x = torch.cartesian_prod(*x)  ## 2^N, N
        return x

    def rejection_sampling_2d(self, x):
        N = self.N
        assert N == 2
        # hexagon
        constraint_fns = {
            lambda x1, x2: x2 + 1. / 3 ** 0.5 * x1 <= 2. / 3 ** 0.5,
            lambda x1, x2: x2 - 1. / 3 ** 0.5 * x1 <= 2. / 3 ** 0.5,
            lambda x1, x2: -x2 + 1. / 3 ** 0.5 * x1 <= 2. / 3 ** 0.5,
            lambda x1, x2: -x2 - 1. / 3 ** 0.5 * x1 <= 2. / 3 ** 0.5,
            lambda x1, x2: 0 * x2 + x1 <= 1,
            lambda x1, x2: 0 * x2 - x1 <= 1,
        }
        B, K = x.shape
        r_max = 2. / 3 ** 0.5
        noise = x.new(B*3, K).normal_(0, 1)
        norm = (noise.pow(2).sum(dim=-1, keepdim=True)) ** 0.5
        r = norm.new(norm.shape).uniform_(0, r_max ** 2) ** 0.5
        noise = r * noise / norm
        noise_per_dim = noise.chunk(N, dim=-1)
        mask = torch.cat([fn(*noise_per_dim) for fn in constraint_fns], dim=-1)
        mask = mask.all(dim=-1)
        noise = noise[mask, :]
        noise = noise[:B, :]
        return noise

    def rejection_sampling(self, num):
        N, r_max = self.N, 1.5 * (self.basis[0][0].item()/2)

        samples = []
        count = 0
        while count < num:
            noise = torch.ones(1000000//N, N).normal_(0, 1).cuda()
            norm = (noise.pow(2).sum(dim=-1, keepdim=True)) ** 0.5
            r = r_max * norm.new(norm.shape).uniform_(0, 1) ** (1. / N)
            noise = r * noise / norm

            lattice_points, _ = self.find_nearest(noise)
            mask = lattice_points.eq(0).all(dim=-1)
            count += mask.sum().item()
            samples.append(noise[mask, :])

        samples = torch.cat(samples, dim=0)[:num, :]
        return samples

    def sphere_samping(self, x):
        B, K = x.shape
        basis, pi = self.basis, torch.pi
        volume = torch.linalg.det(basis).item()
        r_max = sphere_radius(volume, K)

        noise = x.new(B, K).normal_(0, 1)
        norm = (noise.pow(2).sum(dim=-1, keepdim=True)) ** 0.5
        # r = norm.new(norm.shape).uniform_(0, r_max ** 2) ** (1. / 2)
        r = r_max * norm.new(norm.shape).uniform_(0, 1) ** (1. / K)
        noise = r * noise / norm
        return noise

    def find_nearest(self, x):
        B, K = x.shape
        N = self._basis.shape[0]

        basis = self.basis.to(x.device)  ## N, K
        grid = self.grid = self.grid.to(x.device)  ## 2^N, N

        u = []
        x_temp = x
        for i in range(N)[::-1]:
            ui = x_temp[:, i:i + 1] / basis[i:i + 1, i:i + 1]  ## B, 1
            ui = ui.floor()
            u.append(ui)
            if i != 0:
                deltai = ui * basis[i:i + 1, :i]  ## B, i-1
                x_temp = x_temp[:, :i] - deltai  ## B, i-1

        dot = torch.cat(u[::-1], dim=1)
        grid = dot.view(B, 1, N) + grid.view(1, -1, N)  ## B, 2^N, N
        code_book = torch.einsum('bjn,nk->bjk', [grid, basis])  ## B, 2^N, K

        dist = l2_dist(x, code_book)  ## B, 2^N
        dist = dist / K

        index = dist.argmin(dim=-1, keepdim=True).detach().long()  ## B, 1
        one_hot = torch.zeros_like(dist)
        one_hot = one_hot.scatter_(-1, index, 1.0)  ## B, 2^N

        x_hat = torch.einsum('bj,bjk->bk', one_hot, code_book)
        return x_hat, dist

    def forward(self, x, training):
        x_hat, dist = self.find_nearest(x)

        if training:
            x_hat = (x_hat - x).detach() + x
            index = torch.randint(0, len(self.voronoi_samples),
                                  x.shape[0:1])

            noise = self.voronoi_samples[index]
            x_hat_noise = x + noise
            x_hat_noise = x_hat # ste
            # x_indexes = x_hat_noise ## 2
            x_indexes = x_hat
            return x_hat_noise, x_indexes, None
        else:
            x_indexes = x_hat
            return x_hat, x_indexes, dist

    def encode(self, x, k, training):
        B, K = x.shape
        N = self._basis.shape[0]

        basis = self.basis.detach().to(x.device) ## N, K
        grid = self.grid = self.grid.to(x.device)  ## 2^N, N

        u = []
        x_temp = x
        for i in range(N)[::-1]:
            ui = x_temp[:, i:i + 1] / basis[i:i + 1, i:i + 1]  ## B, 1
            ui = ui.floor()
            u.append(ui)
            if i != 0:
                deltai = ui * basis[i:i + 1, :i]  ## B, i-1
                x_temp = x_temp[:, :i] - deltai  ## B, i-1

        dot = torch.cat(u[::-1], dim=1)
        grid = dot.view(B, 1, N) + grid.view(1, -1, N)  ## B, 2^N, N
        code_book = torch.einsum('bjn,nk->bjk', [grid, basis])  ## B, 2^N, K

        dist = l2_dist(x, code_book)  ## B, 2^N
        dist = dist / K

        ## topk
        _, index_topk = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
        index_topk = index_topk.chunk(k, dim=-1)
        x_hat_topk = []
        for i in range(k):
            one_hot = torch.zeros_like(dist)
            one_hot = one_hot.scatter_(-1, index_topk[i], 1.0)  ## B, 2^N
            x_hat_topk.append(torch.einsum('bj,bjk->bk', one_hot, code_book))

        if training:
            x_hat_topk = [(x_hat - x).detach() + x for x_hat in x_hat_topk]
            if self.noise_type == 'ste':
                noise = 0
            elif self.noise_type == 'original':
                noise = self.rejection_sampling(x)
            elif self.noise_type == 'sphere':
                noise = self.sphere_samping(x)
            return x_hat_topk, [x + noise for _ in range(k)], None
            # return x_hat_topk, x_hat_topk, None
        else:
            return x_hat_topk, x_hat_topk, dist


