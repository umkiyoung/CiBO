from models import *
import torch
from gfn_folder.gfn_train import *
import math

class DiffusionSampler:
    def __init__(self, energy, prior, gfn_sampler, buffer, buffer_ls, device, dtype, batch_size, args, beta=1):
        self.energy = energy
        self.prior = prior
        self.sampler = gfn_sampler
        self.beta = beta
        self.bsz = batch_size
        self.buffer = buffer
        self.buffer_ls = buffer_ls
        self.args = args
        self.device = device
        self.dtype = dtype
        
    def train(self, i):
        loss = train_step(
            self.energy,
            self.sampler,
            i,
            self.args.exploratory,
            self.buffer,
            self.buffer_ls,
            self.args.exploration_factor,
            self.args.exploration_wd,
            self.args,
            self.device,
            self.dtype,
        )
        return loss

    def sample(self, batch_size, track_gradient):
        """
        Directly sample from z ~ sampler, return f(z)
        """
        z = self.sampler.sample(batch_size, self.energy.log_reward)           
        x = self.prior.sample_with_noise(z, track_gradient)
        return x

class Energy():
    """
    Directly compute the r(f(z))
    """
    def __init__(self, proxy, prior, alpha, beta):
        self.proxy = proxy
        self.prior = prior
        self.beta = beta
        self.alpha = alpha
        
    def log_reward(self, z, track_gradient=False):
        reward = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1) * self.alpha
        reward += self.proxy.log_reward(self.prior.sample_with_noise(z, track_gradient), beta=self.beta)      
        return reward