from copy import deepcopy 
import math
import time
import einops
import numpy as np
import torch
from data_compression.distributions.uniform_noised import (NoisyDeepFactorized,
                                                           NoisyNormal)
from data_compression.entropy_models import (
    ContinuousConditionalEntropyModel, ContinuousUnconditionalEntropyModel)
from data_compression.quantization import UniformNoise, UniformQuantization
from einops import rearrange, repeat
from torch import nn
from modules import PatchEmbed, PatchUnEmbed, GroupBasicLayer


class GroupTransformerBasedTransformCodingHyper(nn.Module):
    def __init__(self,
                 in_channel=3, out_channel=3,
                 embed_dim_g=[128, 192, 256, 320], depths_g=[2, 2, 5, 1],
                 num_heads_g=[8, 12, 16, 16],
                 embed_dim_h=[192, 192], depths_h=[5, 1], num_heads_h=[12, 12],
                 window_size_g=8, window_size_h=4,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,  # default: True
                 use_checkpoint=False):
        super().__init__()

        self.patch_norm = patch_norm

        # . encoder
        # transform
        self.g_a_pe = nn.ModuleList()  # pe = patch embed
        self.g_a_m = nn.ModuleList()
        in_channels = [in_channel] + embed_dim_g[:-1]
        for i in range(len(embed_dim_g)):
            patchembed = PatchEmbed(
                patch_size=2, in_chans=in_channels[i], embed_dim=embed_dim_g[i],
                norm_layer=norm_layer if self.patch_norm else None)
            self.g_a_pe.append(patchembed)

            layer = GroupBasicLayer(
                dim=embed_dim_g[i],
                depth=depths_g[i],
                num_heads=num_heads_g[i],
                window_size=window_size_g,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.g_a_m.append(layer)

        # hyper - conv based
        y_channel = embed_dim_g[-1]
        embed_dim_h = embed_dim_h + [embed_dim_h[-1]]
        m = []
        for i in range(len(embed_dim_h)):
            Ci = embed_dim_h[i]
            if i == 0:
                m.append(nn.Conv2d(y_channel, Ci, 3, 1, 1))
            else:
                Cim1 = embed_dim_h[i - 1]
                m.append(nn.ReLU())
                m.append(nn.Conv2d(Cim1, Ci, 5, 2, 2))
        self.h_a_m = nn.Sequential(*m)

        # .decoder
        # transform
        self.g_s_m = nn.ModuleList()
        self.g_s_pe = nn.ModuleList()  # pe = patch embed
        out_channels = embed_dim_g[::-1][1:] + [out_channel]
        for i in range(len(embed_dim_g)):
            layer = GroupBasicLayer(
                dim=embed_dim_g[::-1][i],
                depth=depths_g[::-1][i],
                num_heads=num_heads_g[::-1][i],
                window_size=window_size_g,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.g_s_m.append(layer)

            norm_ = norm_layer if self.patch_norm else None
            # if i == len(embed_dim_g) - 1:   # todo : should there be a norm layer for the final un-sampling?
            #     norm_ = None
            patchembed = PatchUnEmbed(
                patch_size=2, in_chans=embed_dim_g[::-1][i], embed_dim=out_channels[i],
                norm_layer=norm_)
            self.g_s_pe.append(patchembed)

        # hyper - conv based
        m = []
        for i in range(len(embed_dim_h))[::-1]:
            Ci = embed_dim_h[i]
            if i == 0:
                m.append(nn.Conv2d(Ci, y_channel * 2, 3, 1, 1))
            else:
                Cim1 = embed_dim_h[i - 1]
                m.append(nn.ConvTranspose2d(Ci, Cim1, 5, 2, 2, 1))
                m.append(nn.ReLU())
        self.h_s_m = nn.Sequential(*m)

        # .entropy model
        scale_min, scale_max, num_scales = 0.11, 256, 64
        offset = math.log(scale_min)
        factor = (math.log(scale_max) - math.log(scale_min)) / (num_scales - 1)
        scale_table = torch.exp(offset + factor * torch.arange(num_scales))
        self.y_em = ContinuousConditionalEntropyModel(
            NoisyNormal, param_tables=dict(loc=[0], scale=scale_table.tolist()))
        self.z_em = ContinuousUnconditionalEntropyModel(
            NoisyDeepFactorized(batch_shape=(embed_dim_h[-1],)))

        self.quant = UniformQuantization(step=1)

    def forward(self, x, noisy=True, keep_bits_batch=False, msk=None, only_rec_fg=False,
                encrypt_msk=None):
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)  # entropy model must use noisy input.

        y_means, y_scales = self.h_s(z_hat).chunk(2, 1)
        _, y_indexes = self.quant(y, offset=y_means, noisy=noisy)  # noisy for entropy estimation
        y_hat, _ = self.quant(y, offset=y_means, noisy=False)  # noisy or ste
        y_loc = torch.zeros(1).to(y.device)
        # bits = self.y_em(y_indexes, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        bits, log_probs = self.y_em(
            y_indexes, draw=True, loc=y_loc, scale=y_scales, keep_batch=keep_bits_batch)
        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b, c, h, w = y.shape
            y_hat[(msk == 0).broadcast_to((b, c, h, w))] = 0

        if encrypt_msk is not None:
            encrypt_msk = torch.from_numpy(encrypt_msk).bool()
            encrypt_msk = einops.repeat(encrypt_msk, 'h w -> b c h w', b=y_hat.shape[0],
                                        c=y_hat.shape[1])
            y_hat[encrypt_msk] = y_hat[encrypt_msk][torch.randperm(y_hat[encrypt_msk].size(0))]

        x_hat = self.g_s(y_hat, msk)

        return {
            "x_hat": x_hat,
            "bits": {"y": bits, "z": side_bits},
            'log_probs': log_probs
        }

    def g_a(self, x, msk):
        for pe, layer in zip(self.g_a_pe, self.g_a_m):
            x = pe(x)
            _, _, Wh, Ww = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww, group_mask=msk)
            x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)
        return x

    def g_s(self, x, msk):
        for pe, layer in zip(self.g_s_pe, self.g_s_m):
            _, _, Wh, Ww = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww, group_mask=msk)
            x = rearrange(x, 'b (h w) c -> b c h w', h=Wh, w=Ww)
            x = pe(x)
        return x

    def h_a(self, x):
        # hyper - conv based
        x = self.h_a_m(x)
        return x

    def h_s(self, x):
        # hyper - conv based
        x = self.h_s_m(x)
        return x

    def init_tables(self):
        for m in self.modules():
            if hasattr(m, '_init_tables'):
                m._init_tables()

    def fix_tables(self):
        for m in self.modules():
            if hasattr(m, '_fix_tables'):
                m._fix_tables()


class GroupChARTTC(GroupTransformerBasedTransformCodingHyper):
    def __init__(self,
                in_channel=3, out_channel=3,
                embed_dim_g=[128, 192, 256, 320], depths_g=[2, 2, 5, 1], num_heads_g=[8, 12, 16, 16],
                embed_dim_h=[192, 192], depths_h=[5, 1], num_heads_h=[12, 12],
                window_size_g=8, window_size_h=4,
                mlp_ratio=4, qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=False,
                use_checkpoint=False,
                splits=10):
        super().__init__(in_channel, out_channel, embed_dim_g,
                        depths_g, num_heads_g, embed_dim_h,
                        depths_h, num_heads_h, window_size_g,
                        window_size_h, mlp_ratio, qkv_bias,
                        qk_scale, norm_layer, patch_norm, use_checkpoint)

        M = embed_dim_g[-1]
        self.splits = splits
        split_channels = M // splits
        assert M % splits == 0
        self.splits = splits
        self.split_channels = split_channels

        self.char_m_means = nn.ModuleList()
        for idx in range(splits):
            self.char_m_means.append(nn.Sequential(
                # nn.Conv2d(split_channels*2 + idx*split_channels, M, 3, 1, 1),
                nn.Conv2d(split_channels*2 + idx*split_channels, M, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(M, M, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(M, split_channels, 1, 1, 0)
            ))
        self.char_m_scales = deepcopy(self.char_m_means)

    def forward(self, x, msk=None, noisy=False, keep_bits_batch=False, only_rec_fg=False, encrypt_msk=None):
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)     # entropy model must use noisy input.
        y_hyper = self.h_s(z_hat)

        # channel autoregressive
        y_slices = y.chunk(self.splits, dim=1)
        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices, bits, log_probs = [], [], []
        for idx in range(self.splits):
            y_slice = y_slices[idx]
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)

            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)
            _, y_indexes = self.quant(y_slice, offset=y_mean, noisy=noisy)
            y_hat_slice, _ = self.quant(y_slice, offset=y_mean, noisy=False)  # STE
            y_loc = torch.zeros(1).to(x.device)
            # slice_bits = self.y_em(y_indexes, loc=y_loc, scale=y_scale)
            slice_bits, slice_log_probs = self.y_em(
                y_indexes, draw=True, loc=y_loc, scale=y_scale)
            bits.append(slice_bits)
            y_hat_slices.append(y_hat_slice)
            log_probs.append(slice_log_probs)
        bits = sum(bits)
        y_hat = torch.cat(y_hat_slices, dim=1)
        log_probs = torch.cat(log_probs, dim=1)

        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0

        if encrypt_msk is not None:
            encrypt_msk = torch.from_numpy(encrypt_msk).bool()
            encrypt_msk = einops.repeat(encrypt_msk, 'h w -> b c h w', b=y_hat.shape[0], c=y_hat.shape[1])
            y_hat[encrypt_msk] = y_hat[encrypt_msk][torch.randperm(y_hat[encrypt_msk].size(0))]

        x_hat = self.g_s(y_hat, msk)

        return x_hat, bits, side_bits
    def compress(self, x, group_mask, reconstruct=False):
        """Compresses an image tensor."""
        y = self.g_a(x, group_mask)
        y = torch.clamp(y, min=-255.5, max=256.49)
        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=False)     # entropy model must use noisy input.
        y_hyper = self.h_s(z_hat)

        group_strings = []
        for group_idx in group_mask.unique():
            if group_idx == -1: # -1 means not compress
                continue
            single_group_mask = group_mask==group_idx
            y_tmp = y.clone()
            group_string, y_hat_slices = self._group_compress_ChARM(y_tmp, y_hyper, single_group_mask)
            group_strings.append(group_string)

        side_string = self.z_em.compress(z_indexes)
        strings = [group_strings, side_string]
        if reconstruct:
            y_hat = torch.cat(y_hat_slices, dim=1)
            x_hat = self.g_s(y_hat, group_mask)
            return strings, x_hat
        else:
            return strings, {}

    def _group_compress_ChARM(self, y, y_hyper, group_mask):
        # channel autoregressive
        string = []
        y_slices = y.chunk(self.splits, dim=1)
        B, cy, H, W = y_slices[0].shape[:]
        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices = []
        for idx in range(self.splits):
            y_slice = y_slices[idx]
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)

            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)

            _, y_indexes = self.quant(y_slice, offset=y_mean, noisy=False)
            y_hat_slice, _ = self.quant(y_slice, offset=y_mean, noisy=False)  # STE

            # compress elements of group for idx-th slice
            n_elements = (group_mask.int()).sum()
            y_scale_group = y_scale.masked_select(group_mask).reshape(cy, n_elements)
            y_loc = torch.zeros(1).to(y.device)
            y_indexes_group = y_indexes.masked_select(group_mask).reshape(cy, n_elements)
            string_group = self.y_em.compress(y_indexes_group, loc=y_loc, scale=y_scale_group)
            string.append(string_group)

            y_hat_slices.append(y_hat_slice)
        return string, y_hat_slices

    def decompress(self, strings, shape, group_mask, group_idxs):
        """Decompresses an image tensor."""
        self.y_em.to_list()
        ZDec, YDec = 0,0
        group_strings, side_string = strings
        factor = 64
        z_shape = [int(math.ceil(s / factor)) for s in shape]
        torch.cuda.synchronize()
        t0 = time.time()
        z_indexes = self.z_em.decompress(side_string, z_shape)
        z_hat = self.quant.dequantize(z_indexes)
        torch.cuda.synchronize()
        ZDec += time.time() - t0
        y_hyper = self.h_s(z_hat)

        torch.cuda.synchronize()
        t0 = time.time()
        y_hat = self._group_decompress_ChARM(group_strings, y_hyper, group_mask, group_idxs)
        torch.cuda.synchronize()
        YDec += time.time() - t0
        x_hat = self.g_s(y_hat, group_mask)
        return x_hat, {'YDec': YDec, 'ZDec':ZDec}

    def _group_decompress_ChARM(self, strings, y_hyper, group_mask, group_idxs):
        B, cy, H, W = y_hyper.shape
        cy = int(cy / 2 / self.splits)

        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices = []
        for idx in range(self.splits):
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)
            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)
            y_hat_slice = torch.zeros((B, cy, H, W)).to(y_hyper.device)

            for i, group_idx in enumerate(group_idxs):
                group_string = strings[i].pop(0) # start from slice 0
                single_group_mask = group_mask==group_idx
                n_elements = (single_group_mask.int()).sum()
                y_scale_group = y_scale.masked_select(single_group_mask).reshape(cy, n_elements)
                y_mean_group = y_mean.masked_select(single_group_mask).reshape(cy, n_elements)
                y_loc = torch.zeros(1).to(y_hyper.device)
                y_indexes_group = self.y_em.decompress(group_string, loc=y_loc, scale=y_scale_group)
                y_hat_group = self.quant.dequantize(y_indexes_group, offset=y_mean_group)
                y_hat_slice[single_group_mask.broadcast_to((B, cy, H, W))] = y_hat_group.float().reshape(-1)

            y_hat_slices.append(y_hat_slice)
        y_hat = torch.cat(y_hat_slices, dim=1)
        return y_hat


class GroupChARTTC_woStructure(GroupTransformerBasedTransformCodingHyper):
    def __init__(self,
                in_channel=3, out_channel=3,
                embed_dim_g=[128, 192, 256, 320], depths_g=[2, 2, 5, 1], num_heads_g=[8, 12, 16, 16],
                embed_dim_h=[192, 192], depths_h=[5, 1], num_heads_h=[12, 12],
                window_size_g=8, window_size_h=4,
                mlp_ratio=4, qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=False,
                use_checkpoint=False,
                splits=10):
        super().__init__(in_channel, out_channel, embed_dim_g,
                        depths_g, num_heads_g, embed_dim_h,
                        depths_h, num_heads_h, window_size_g,
                        window_size_h, mlp_ratio, qkv_bias,
                        qk_scale, norm_layer, patch_norm, use_checkpoint)

        M = embed_dim_g[-1]
        self.splits = splits
        split_channels = M // splits
        assert M % splits == 0
        self.splits = splits
        self.split_channels = split_channels

        self.char_m_means = nn.ModuleList()
        for idx in range(splits):
            self.char_m_means.append(nn.Sequential(
                # nn.Conv2d(split_channels*2 + idx*split_channels, M, 3, 1, 1),
                nn.Conv2d(split_channels*2 + idx*split_channels, M, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(M, M, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(M, split_channels, 1, 1, 0)
            ))
        self.char_m_scales = deepcopy(self.char_m_means)

    def forward(self, x, msk=None, noisy=False, keep_bits_batch=False, only_rec_fg=False, encrypt_msk=None):
        y = self.g_a(x, msk)
        y = torch.clamp(y, min=-255.5, max=256.49)

        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=noisy)
        y_hyper = self.h_s(z_hat)

        # channel autoregressive
        y_slices = y.chunk(self.splits, dim=1)
        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices, bits, log_probs = [], [], []
        for idx in range(self.splits):
            y_slice = y_slices[idx]
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)

            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)
            _, y_indexes = self.quant(y_slice, offset=y_mean, noisy=noisy)
            y_hat_slice, _ = self.quant(y_slice, offset=y_mean, noisy=False)  # STE
            y_loc = torch.zeros(1).to(x.device)
            # slice_bits = self.y_em(y_indexes, loc=y_loc, scale=y_scale)
            slice_bits, slice_log_probs = self.y_em(
                y_indexes, draw=True, loc=y_loc, scale=y_scale)
            bits.append(slice_bits)
            y_hat_slices.append(y_hat_slice)
            log_probs.append(slice_log_probs)
        bits = sum(bits)
        y_hat = torch.cat(y_hat_slices, dim=1)
        log_probs = torch.cat(log_probs, dim=1)

        side_bits = self.z_em(z_indexes, keep_batch=keep_bits_batch)

        if only_rec_fg:
            b,c,h,w = y.shape
            y_hat[(msk==0).broadcast_to((b,c,h,w))] = 0

        if encrypt_msk is not None:
            encrypt_msk = torch.from_numpy(encrypt_msk).bool()
            encrypt_msk = einops.repeat(encrypt_msk, 'h w -> b c h w', b=y_hat.shape[0], c=y_hat.shape[1])
            y_hat[encrypt_msk] = y_hat[encrypt_msk][torch.randperm(y_hat[encrypt_msk].size(0))]

        x_hat = self.g_s(y_hat, msk)

        return x_hat, bits, side_bits

    def compress(self, x, group_mask, reconstruct=False):
        """Compresses an image tensor."""
        y = self.g_a(x, group_mask)
        y = torch.clamp(y, min=-255.5, max=256.49)
        z = self.h_a(y)
        z_hat, z_indexes = self.quant(z, noisy=False)
        y_hyper = self.h_s(z_hat)

        group_string, y_hat_slices = self._group_compress_ChARM(y, y_hyper)
        side_string = self.z_em.compress(z_indexes)
        strings = [group_string, side_string]
        if reconstruct:
            y_hat = torch.cat(y_hat_slices, dim=1)
            x_hat = self.g_s(y_hat, group_mask)
            return strings, x_hat
        else:
            return strings, {}

    def _group_compress_ChARM(self, y, y_hyper):
        # channel autoregressive
        string = []
        y_slices = y.chunk(self.splits, dim=1)
        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices = []
        for idx in range(self.splits):
            y_slice = y_slices[idx]
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)

            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)

            _, y_indexes = self.quant(y_slice, offset=y_mean, noisy=False)
            y_hat_slice, _ = self.quant(y_slice, offset=y_mean, noisy=False)  # STE

            y_loc = torch.zeros(1).to(y.device)
            string_slice = self.y_em.compress(y_indexes, loc=y_loc, scale=y_scale)
            string.append(string_slice)
            y_hat_slices.append(y_hat_slice)
        return string, y_hat_slices

    def decompress(self, strings, shape, group_mask=None):
        """Decompresses an image tensor."""
        self.y_em.to_list()
        ZDec, YDec = 0,0
        group_strings, side_string = strings
        factor = 64
        z_shape = [int(math.ceil(s / factor)) for s in shape]
        torch.cuda.synchronize()
        t0 = time.time()
        z_indexes = self.z_em.decompress(side_string, z_shape)
        z_hat = self.quant.dequantize(z_indexes)
        torch.cuda.synchronize()
        ZDec += time.time() - t0
        y_hyper = self.h_s(z_hat)

        torch.cuda.synchronize()
        t0 = time.time()
        y_hat = self._group_decompress_ChARM(group_strings, y_hyper)
        torch.cuda.synchronize()
        YDec += time.time() - t0
        x_hat = self.g_s(y_hat, group_mask)
        return x_hat, {'YDec': YDec, 'ZDec':ZDec}

    def _group_decompress_ChARM(self, strings, y_hyper):
        B, cy, H, W = y_hyper.shape
        cy = int(cy / 2 / self.splits)
        y_hyper_slices = y_hyper.chunk(self.splits, dim=1)
        y_hat_slices = []

        for idx in range(self.splits):
            y_hyper_slice = y_hyper_slices[idx]
            y_context_slices = y_hat_slices[:idx]
            support = torch.cat([y_hyper_slice] + y_context_slices, dim=1)

            y_mean = self.char_m_means[idx](support)
            y_scale = self.char_m_scales[idx](support)

            y_loc = torch.zeros(1).to(y_hyper.device)
            y_indexes = self.y_em.decompress(strings.pop(0), loc=y_loc,
                                                   scale=y_scale)
            y_hat_slice = self.quant.dequantize(y_indexes, offset=y_mean)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        return y_hat
