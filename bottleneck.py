# Author: fanqiNO1
# Date: 2022-05-20
# Description:
# Based on the https://github.com/neuralchen/SimSwap/blob/main/models/fs_networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, latent_size=512):
        super(ResnetBlock, self).__init__()
        conv_1 = []
        conv_1 += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), InstanceNorm()]
        self.conv_1 = nn.Sequential(*conv_1)
        self.style_1 = ApplyStyle(latent_size, dim)
        self.act_1 = nn.ReLU()

        conv_2 = []
        conv_2 += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), InstanceNorm()]
        self.conv_2 = nn.Sequential(*conv_2)
        self.style_2 = ApplyStyle(latent_size, dim)
        self.act_2 = nn.ReLU()

    def forward(self, x, dlatents_in_slice):
        y = self.conv_1(x)
        y = self.style_1(y, dlatents_in_slice)
        y = self.act_1(y)
        y = self.conv_2(y)
        y = self.style_2(y, dlatents_in_slice)
        y = self.act_2(y)
        result = x + y

        return result

class Bottleneck(nn.Module):
    def __init__(self, dim, latent_size=512, n_blocks=6):
        super(Bottleneck, self).__init__()
        bottleneck = []
        for i in range(n_blocks):
            bottleneck += [ResnetBlock(dim, latent_size)]
        self.bottleneck = nn.Sequential(*bottleneck)

    def forward(self, x, dlatents_in_slice):
        for i in range(len(self.bottleneck)):
            x = self.bottleneck[i](x, dlatents_in_slice)
        return x