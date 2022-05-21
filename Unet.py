# Author: fanqiNO1
# Date: 2022-05-18
# Description:
# Based on the https://nn.labml.ai/diffusion/ddpm/unet.html

import torch
import torch.nn as nn
import torch.nn.functional as F

from Unet_part import SiLU, TimeEmbedding, ResidualBlock, DownBlock, UpBlock, MiddleBlock, Downsample, Upsample

class Unet(nn.Module):
    def __init__(self, image_channels=3, n_channels=64, ch_mults=(1, 2, 2, 4), is_attn=(False, False, True, True), n_blocks=2):
        super(Unet, self).__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, n_channels * 4)

        up = []
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = SiLU()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x, t, latent_id):
        t = self.time_emb(t)

        x = self.image_proj(x)
        h = [x]
        for down_i in self.down:
            x = down_i(x, t)
            h.append(x)

        x = self.middle(x, t, latent_id)
        for up_i in self.up:
            if isinstance(up_i, Upsample):
                x = up_i(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = up_i(x, t)

        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    model = Unet()
    x = torch.zeros(1, 3, 224, 224)
    t = x.new_full((1,), 1)
    latent_id = torch.zeros(1, 512)
    with open("Unet.txt", "w") as f:
        results = str(summary(model, input_data=[x, t, latent_id], device='cpu', depth=10))
        f.write(results)