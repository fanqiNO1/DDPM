# Author: fanqiNO1
# Date: 2022-05-20
# Description:
# Based on the https://nn.labml.ai/diffusion/ddpm/

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn

class DenoiseDiffusion:
    def __init__(self, eps_model, n_steps, device):
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, self.n_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    @staticmethod
    def gather(consts, t):
        c = consts.gather(-1, t)
        c = c.reshape(-1, 1, 1, 1)
        return c
    
    def q_xt_x0(self, x0, t):
        mean = DenoiseDiffusion.gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - DenoiseDiffusion.gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt, t):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = DenoiseDiffusion.gather(self.alpha_bar, t)
        alpha = DenoiseDiffusion.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = DenoiseDiffusion.gather(self.sigma2, t)
        eps = torch.randn(xt.shape).to(self.device)
        return mean + (var ** 0.5) * eps

    def loss(self, x0, noise=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,)).to(self.device)
        if noise is None:
            noise = torch.randn_like(x0)
 
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)