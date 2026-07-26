import copy

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from torch.nn import functional as F


class DiffOPT:
    def __init__(self, prior_model, proxy_model_ens, alpha, beta, dtype, device):
        self.prior_model = prior_model
        self.proxy_model_ens = proxy_model_ens
        self.alpha = alpha
        self.beta = beta
        self.dtype = dtype
        self.device = device
        
        self.prior_model.to(device)
        self.proxy_model_ens.to(device)
        self.x_dim = prior_model.x_dim
        
    def sample(self, bs, device):
        # Sample from the prior model with warmup by guiding the model with the proxy model
        normal_dist = torch.distributions.Normal(
            torch.zeros((bs, self.x_dim), device=device, dtype=self.dtype), 
            torch.ones((bs, self.x_dim), device=device, dtype=self.dtype)
        )
        x = normal_dist.sample()
        t = torch.zeros((bs,), device=device, dtype=self.dtype)
        dt = 1 / self.prior_model.diffusion_steps
        
        for i in range(self.prior_model.diffusion_steps):
            bc_epsilon = self.prior_model(x, t)
            q_epsilon = -self.proxy_model_ens.score(x, beta=self.beta)
            epsilon = bc_epsilon + q_epsilon
            
            new_x = self.prior_model.oneover_sqrta[i] * (
                x - self.prior_model.mab_over_sqrtmab_inv[i] * epsilon
            ) + torch.sqrt(self.prior_model.beta_t[i]) * torch.randn_like(x, dtype=self.dtype, device=device)
            
            x = new_x
            t = t + dt

        # MH Correction
        for i in range(self.prior_model.diffusion_steps):
            t = torch.ones((bs,), device=device, dtype=self.dtype)
            
            q_epsilon1 = self.proxy_model_ens.score(x, beta=1.0)
            new_x = self.prior_model.oneover_sqrta[i] * (
                x - self.prior_model.mab_over_sqrtmab_inv[i] * q_epsilon1
            ) + torch.sqrt(self.prior_model.beta_t[i]) * torch.randn_like(x, dtype=self.dtype, device=device)
            q_epsilon2 = self.proxy_model_ens.score(new_x, beta=1.0)
            
            u = torch.rand(bs, device=device, dtype=self.dtype)
            
            bc_epsilon1 = self.prior_model(x, t)
            l1 = -0.5 * torch.norm(bc_epsilon1, dim=1) ** 2
            
            bc_epsilon2 = self.prior_model(new_x, t)
            l2 = -0.5 * torch.norm(bc_epsilon2, dim=1) ** 2
            l1tol2 = -torch.norm(x - new_x - dt * (bc_epsilon2 - q_epsilon2), dim=1) ** 2
            l2tol1 = -torch.norm(new_x - x - dt * (bc_epsilon1 - q_epsilon1), dim=1) ** 2
            log_accept = l2 - l1 + (l1tol2 - l2tol1) / (2.0 * dt)
            
            x[torch.log(u) < log_accept] = new_x[torch.log(u) < log_accept]
            
        return x
            