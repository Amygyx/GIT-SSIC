import torch
import torch.nn as nn
import torch.nn.functional as F

from .gdn import GDN

# __all__ = ['ResBlock', 'ResBlocks']


# class ResBlock(nn.Module):
#     def __init__(self, c, ks=3):
#         super().__init__()
#         self.m = nn.Sequential(
#             nn.Conv2d(c, c, ks, 1, ks//2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c, c, ks, 1, ks//2)
#         )
#     def forward(self, x):
#         return x + self.m(x)


# class ResBlocks(nn.Module):
#     def __init__(self, c, n=3, ks=3):
#         super().__init__()
#         self.m = nn.Sequential(
#             *([ResBlock(c, ks) for _ in range(n)])
#         )
#     def forward(self, x):
#         print("{}".format(torch.cuda.memory_allocated(0)))
#         return x + self.m(x)


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, inplace=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out += identity
        return out

class ResBlocks(nn.Module):
    def __init__(self, c, n=3):
        super().__init__()
        self.m = nn.Sequential(
            *([ResBlock(c, c) for _ in range(n)])
        )
    def forward(self, x):
        return x + self.m(x)


class Downsample(nn.Module):

    def __init__(self, c_in, c_out, factor=2):
        super().__init__()
        self.m = nn.Sequential(
            nn.PixelUnshuffle(factor),
            nn.Conv2d(c_in * factor ** 2, c_out, 1, 1, 0),
        )

    def forward(self, x):
        return self.m(x)


class Upsample(nn.Module):

    def __init__(self, c_in, c_out, factor=2):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(c_in, c_out * factor ** 2, 1, 1, 0),
            nn.PixelShuffle(factor),
        )

    def forward(self, x):
        return self.m(x)

class ResBlockDown(nn.Module):
    def __init__(self, inplanes, outplanes, inplace=False):
        super(ResBlockDown, self).__init__()
        self.conv0 = Downsample(inplanes, outplanes)
        if inplanes == 3:
            self.conv1 = Downsample(3, outplanes)
            self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)            
        else:
            self.conv1 = Downsample(inplanes, outplanes)
            self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=inplace)
        self.gdn = GDN(outplanes, inverse=False)

    def forward(self, x):
        identity = self.conv0(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out += identity
        return out

class ResBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, inplace=False):
        super(ResBlockUp, self).__init__()
        self.conv0 = Upsample(inplanes, outplanes)
        self.conv1 = Upsample(inplanes, inplanes)
        self.relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.gdn = GDN(outplanes, inverse=False)

    def forward(self, x):
        identity = self.conv0(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out += identity
        return out

class ResBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, groups, inplace=False):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes // 2, kernel_size=1, stride=1, groups=groups)
        self.conv2 = nn.Conv2d(inplanes // 2, inplanes // 2, kernel_size=3, stride=1, padding=1, groups=groups)
        self.conv3 = nn.Conv2d(inplanes // 2, outplanes, kernel_size=1, stride=1, groups=groups)
        self.relu = nn.LeakyReLU(inplace=inplace)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out += identity
        return out

class ResAttn(nn.Module):
    def __init__(self, inplanes, outplanes, inplace=False):
        super(ResAttn, self).__init__()
        self.query_conv = nn.Sequential(
            ResBottleneck(inplanes, inplanes, groups=1, inplace=inplace),
            ResBottleneck(inplanes, inplanes, groups=1, inplace=inplace),
            ResBottleneck(inplanes, inplanes, groups=1, inplace=inplace),
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.key_conv = nn.Sequential(
            ResBottleneck(inplanes, inplanes, groups=1, inplace=inplace),
            ResBottleneck(inplanes, inplanes, groups=1, inplace=inplace),
            ResBottleneck(inplanes, inplanes, groups=1, inplace=inplace)
        )     
        self.gamma = nn.Parameter(torch.zeros(1))   

    def forward(self, x):
        identity = x
        query_out = self.query_conv(x)
        key_out = self.key_conv(x)
        out = identity + self.gamma * (query_out * key_out)
        return out

class ResAttnSplit(nn.Module):
    def __init__(self, inplanes, outplanes, groups, inplace=False):
        super(ResAttnSplit, self).__init__()
        self.groups = groups
        self.split_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes * groups, kernel_size=3, stride=1, padding=1, groups=groups),
            ResBottleneck(outplanes * groups, outplanes * groups, groups, inplace),
            ResBottleneck(outplanes * groups, outplanes * groups, groups, inplace),
            ResBottleneck(outplanes * groups, outplanes * groups, groups, inplace)
        )
        self.fc1 = nn.Conv2d(outplanes, outplanes // 4, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(outplanes // 4, outplanes * groups, kernel_size=1, stride=1)
        self.last_conv = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1)

    def forward(self, x):
        x_group = self.split_conv(x)
        batch, channel = x_group.shape[:2]
        if self.groups > 1:
            splited = torch.split(x_group, channel // self.groups, dim=1)
            gap = sum(splited) 
        else:
            gap = x_group
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap).view((batch, self.groups, channel // self.groups))
        if self.groups > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten).view(batch, -1, 1, 1)
        if self.groups > 1:
            atten = torch.split(atten, channel // self.groups, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        out = x + self.last_conv(out)
        return out