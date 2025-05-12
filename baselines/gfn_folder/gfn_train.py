import argparse
import torch
import os

from baselines.gfn_folder.gfn_utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name
from baselines.gfn_folder.buffer import ReplayBuffer
from baselines.gfn_folder.langevin import langevin_dynamics
from baselines.models.gfn import GFN
from baselines.gfn_folder.gfn_losses import *
from baselines.gfn_folder.evaluations import *

import matplotlib.pyplot as plt
from tqdm import trange
import wandb


def get_exploration_std(iter, exploratory, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - iter / 5000.)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl

def train_step(energy, gfn_model, it, exploratory, buffer, buffer_ls, exploration_factor, exploration_wd, args, device, dtype):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, exploration_std, args, device, return_exp=True, dtype=dtype)
                X = states[:, -1]                    
                buffer.add(X, log_r)
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std, args, device, dtype=dtype)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, args, device, exploration_std, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, args, device,exploration_std, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, exploration_std,args,device, dtype=dtype)


    return loss

def eval_step(eval_data, energy, gfn_model, final_eval=False, final_eval_data_size=None, eval_data_size = None, device = "cpu", args=None):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z'], metrics['final_eval/log_Z_lb'], metrics[
            'final_eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                              gfn_model,
                                                                                                              energy.log_reward)
        else:
            metrics['eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                        gfn_model,
                                                                                                        energy.log_reward)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics


def fwd_train_step(energy, gfn_model, exploration_std, args, device, return_exp=False, dtype=torch.float32):
    init_state = torch.zeros(args.gfn_batch_size, args.dim, dtype=dtype).to(device=device, dtype=dtype)
    coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device=device, dtype=dtype)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss

def bwd_train_step(energy, gfn_model, buffer, buffer_ls, args, device, exploration_std=None,it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.gfn_batch_size, exploration_std).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.gfn_batch_size).to(device)
    elif args.sampling == 'buffer':
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, rewards = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer_ls.add(local_search_samples, log_r)
        
            samples, rewards = buffer_ls.sample()
        else:
            samples, rewards = buffer.sample()

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward,
                                exploration_std=exploration_std)
    return loss
def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, coeff_matrix, exploration_std=None, return_exp=False):
    if mode == 'tb':
        loss = fwd_tb(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif mode == 'tb-avg':
        loss = fwd_tb_avg(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif mode == 'db':
        loss = db(init_state, gfn_model, log_reward, exploration_std)
    elif mode == 'subtb':
        loss = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std)
    return loss

def get_gfn_optimizer(gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False, use_weight_decay=False, weight_decay=1e-7):
    param_groups = [ {'params': gfn_model.t_model.parameters()},
                    {'params': gfn_model.s_model.parameters()},
                    {'params': gfn_model.joint_model.parameters()},
                    {'params': gfn_model.langevin_scaling_model.parameters()} ]
    if conditional_flow_model:
        param_groups += [ {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow} ]
    else:
        param_groups += [ {'params': [gfn_model.flow_model], 'lr': lr_flow} ]

    if back_model:
        param_groups += [ {'params': gfn_model.back_model.parameters(), 'lr': lr_back} ]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer
