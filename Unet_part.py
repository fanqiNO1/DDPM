# Author: fanqiNO1
# Date: 2022-05-18
# Description:
# Based on the https://nn.labml.ai/diffusion/ddpm/unet.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return F.sigmoid(x) * x


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super(TimeEmbedding, self).__init__()
        self.n_channels = n_channels
        self.linear_1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.activation = SiLU()
        self.linear_2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels // 8
        emb = -torch.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        emb = self.linear_1(emb)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups=32):
        super(ResidualBlock, self).__init__()
        self.norm_1 = nn.GroupNorm(n_groups, in_channels)
        self.activation_1 = SiLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm_2 = nn.GroupNorm(n_groups, out_channels)
        self.activation_2 = SiLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        h = self.norm_1(x)
        h = self.activation_1(h)
        h = self.conv_1(h)
        h += self.time_emb(t)[:, :, None, None]
        h = self.norm_2(h)
        h = self.activation_2(h)
        h = self.conv_2(h)
        h += self.shortcut(x)
        return h

class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None, n_groups=32):
        super(AttentionBlock, self).__init__()
        if d_k is None:
            d_k = n_channels
        
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t=None):
        