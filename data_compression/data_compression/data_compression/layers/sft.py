import torch
import torch.nn as nn
import torch.nn.functional as F

class QmapFuse(nn.Module):
    def __init__(self, c_in, c_out, channel_expand=2):
        super().__init__()
        m = []
        for i in range(channel_expand):
            if i == 0:
                m.append(nn.Conv2d(c_in, c_out * 2 ** channel_expand, 3, 1, 1))
                m.append(nn.LeakyReLU(0.1, inplace=True))
            else:
                m.append(nn.Conv2d(c_out * 2 ** (channel_expand - i + 1), c_out * 2 ** (channel_expand - i), 3, 1, 1))
                m.append(nn.LeakyReLU(0.1, inplace=True))
        m.append(nn.Conv2d(c_out * 2, c_out, 3, 1, 1))
        self.fuse = nn.Sequential(*m)
    
    def forward(self, x):
        return self.fuse(x)


class QmapDownsample(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.m(x)


class QmapUpsample(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.m = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.m(x)


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap, qmap_msk=None):
        if qmap_msk != None:
            qmap_msk = F.interpolate(qmap_msk, x.size()[2:], mode='nearest')
        else:
            qmap_msk = torch.ones_like(x).mean(dim=1, keepdim=True)
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap) * qmap_msk
        gamma = self.mlp_gamma(actv) * qmap_msk
        beta = self.mlp_beta(actv) * qmap_msk
        out = (x * (1 + gamma) + beta) * qmap_msk
        return out


class SFTResblk(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)

        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    
    def forward(self, x, qmap, qmap_msk=None):
        if qmap_msk != None:
            qmap_msk = F.interpolate(qmap_msk, x.size()[2:], mode='nearest')
        else:
            qmap_msk = torch.ones_like(x).mean(dim=1, keepdim=True)
        dx = self.conv_0(self.actvn(self.norm_0(x, qmap, qmap_msk))) * qmap_msk
        dx = self.conv_1(self.actvn(self.norm_1(dx, qmap, qmap_msk))) * qmap_msk
        out = (x + dx) * qmap_msk

        return out

