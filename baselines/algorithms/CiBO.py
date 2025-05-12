import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from tqdm import trange
import torch
from functions.test_function import TestFunction
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.value_functions import ProxyEnsemble
from models.flow import FlowModel
from models.gfn import GFN
from models.diffusion_sampler import DiffusionSampler, Energy
from gfn_folder.gfn_utils import get_gfn_optimizer
from gfn_folder.buffer import ReplayBuffer
from utils import save_numpy_array, set_seed, get_lagrangian
import wandb
import multiprocessing as mp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for the Black-box optimization
    parser.add_argument("--task", type=str, default="Ackley")
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_evals", type=int, default=10000)
    parser.add_argument("--dtype", type=str, default="float32", choices = ["float32", "float64"])

    # Arguments for CiBO Methods
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--num_proxy_epochs", type=int, default=100)
    parser.add_argument("--num_prior_epochs", type=int, default=100)
    parser.add_argument("--num_posterior_epochs", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--flow_steps", type=int, default=30) # Flow integration steps
    parser.add_argument("--proxy_hidden_dim", type=int, default=256)
    parser.add_argument("--prior_hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--M", type=int, default=10) # Number of samples for filtering
    parser.add_argument("--filtering", type=str, default="false", choices=('true', 'false'))
    parser.add_argument("--num_ensembles", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./baselines/results/CiBO/")
    parser.add_argument("--wandb", type=str, default="false", choices=('true', 'false'))
    parser.add_argument("--indicator", type=str, default="false", choices=('true', 'false'))

    ################################################################
    # Arguments for diffusion sampler mostly from Improved Offpolicy Training of Diffusion Sampler
    parser.add_argument('--lr_policy', type=float, default=1e-3)
    parser.add_argument('--lr_flow', type=float, default=1e-2)
    parser.add_argument('--lr_back', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--s_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--harmonics_dim', type=int, default=64)
    parser.add_argument('--gfn_batch_size', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=25000)
    parser.add_argument('--gfn_buffer_size', type=int, default=300 * 1000 * 2)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--subtb_lambda', type=int, default=2)
    parser.add_argument('--t_scale', type=float, default=5.)
    parser.add_argument('--log_var_range', type=float, default=4.)
    parser.add_argument('--energy', type=str, default='9gmm',
                        choices=('9gmm', '25gmm', 'hard_funnel', 'easy_funnel', 'many_well'))
    parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
    parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
    parser.add_argument('--both_ways', action='store_true', default=False)

    # For local search
    ################################################################
    parser.add_argument('--local_search', action='store_true', default=False)
    # How many iterations to run local search
    parser.add_argument('--max_iter_ls', type=int, default=200)
    # How many iterations to burn in before making local search
    parser.add_argument('--burn_in', type=int, default=100)
    # How frequently to make local search
    parser.add_argument('--ls_cycle', type=int, default=100)
    # langevin step size
    parser.add_argument('--ld_step', type=float, default=0.001)
    parser.add_argument('--ld_schedule', action='store_true', default=False)
    # target acceptance rate
    parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


    # For replay buffer
    ################################################################
    # high beta give steep priorization in reward prioritized replay sampling
    # parser.add_argument('--beta', type=float, default=1.)

    # low rank_weighted give steep priorization in rank-based replay sampling
    parser.add_argument('--rank_weight', type=float, default=1e-2)
    # three kinds of replay training: random, reward prioritized, rank-based
    parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'exp', 'rank'))
    ################################################################
    parser.add_argument('--bwd', action='store_true', default=False)
    parser.add_argument('--exploratory', action='store_true', default=False)
    parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
    parser.add_argument('--langevin', action='store_true', default=False)  
    parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
    parser.add_argument('--conditional_flow_model', action='store_true', default=False)
    parser.add_argument('--learn_pb', action='store_true', default=False)
    parser.add_argument('--pb_scale_range', type=float, default=0.1)
    parser.add_argument('--learned_variance', action='store_true', default=False)
    parser.add_argument('--partial_energy', action='store_true', default=False)
    parser.add_argument('--exploration_factor', type=float, default=0.1)
    parser.add_argument('--exploration_wd', action='store_true', default=False)
    parser.add_argument('--clipping', action='store_true', default=False)
    parser.add_argument('--lgv_clip', type=float, default=1e2)
    parser.add_argument('--gfn_clip', type=float, default=1e4)
    parser.add_argument('--zero_init', action='store_true', default=False)
    parser.add_argument('--pis_architectures', action='store_true', default=False)
    parser.add_argument('--lgv_layers', type=int, default=3)
    parser.add_argument('--joint_layers', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--use_weight_decay', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    ################################################################
    if args.wandb == "true":
        wandb.init(project="CiBO",
                config=vars(args))
    
    task = args.task
    dim = args.dim
    batch_size = args.batch_size
    n_init = args.n_init
    seed = args.seed
    indicator = args.indicator
    lamb = args.lamb
    train_batch_size = args.train_batch_size
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(seed)
    test_function = TestFunction(task = task, dim = dim, n_init = n_init, seed = seed, indicator=indicator, dtype=dtype, device=device)
    test_function.get_initial_points()
   
    num_rounds = (args.max_evals - n_init) // batch_size
    num_proxy_epochs = args.num_proxy_epochs
    num_prior_epochs = args.num_prior_epochs
    num_posterior_epochs = args.num_posterior_epochs
    
    X_total = test_function.X.cpu().numpy()
    Y_total = test_function.Y.cpu().numpy()
    C_total = test_function.C.cpu().numpy()
    True_total = test_function.true_score.cpu().numpy()

    print(test_function.Y.shape)
    print(test_function.C.shape)
    
    dim_c = test_function.C.shape[1]
    
    for round in range(num_rounds):
        start_time = time.time()
        test_function.X_mean = test_function.X.mean(dim=0)
        test_function.X_std = test_function.X.std(dim=0)
        
        test_function.Y_mean = test_function.Y.mean()
        test_function.Y_std = test_function.Y.std()
        
        test_function.C_mean = test_function.C.mean(dim=0)
        test_function.C_std = test_function.C.std(dim=0)
        
        norm_Y = (test_function.Y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
        norm_C = test_function.normalizing_constraints(test_function.C)
        
        weights = get_lagrangian(norm_Y.squeeze(), norm_C, lamb, indicator)
        weights = torch.exp(weights)

        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)    
        data_loader = DataLoader(test_function, batch_size=train_batch_size, sampler=sampler)
        
        proxy_model_ens = ProxyEnsemble(x_dim=dim, hidden_dim=args.proxy_hidden_dim, output_dim=1 + dim_c, 
                                        num_hidden_layers=3, n_ensembles=args.num_ensembles, ucb_reward=True, indicator=args.indicator,
                                        lamb=lamb).to(dtype=dtype, device=device)
        proxy_model_ens.gamma = args.gamma

        #---------------------------------------------------------------------------
        # Proxy training part
        proxy_model_optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-3) 
            for model in proxy_model_ens.models
        ]

        constraint_model_optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-3) 
            for model in proxy_model_ens.constraint_models
        ]

        for epoch in tqdm(range(num_proxy_epochs), dynamic_ncols=True):
            total_loss = 0.0
            for x, y, c in data_loader:
                x_norm = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
                x_norm += torch.randn_like(x_norm) * 0.001
                y_norm = (y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
                
                # Proxy model optimization
                for model, optimizer in zip(proxy_model_ens.models, proxy_model_optimizers):
                    optimizer.zero_grad()
                    loss = model.compute_loss(x_norm, y_norm)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Constraint model optimization
                c_norm = test_function.normalizing_constraints(c)  # Adjust as necessary
                for j, (model, optimizer) in enumerate(zip(proxy_model_ens.constraint_models, constraint_model_optimizers)):
                    optimizer.zero_grad()
                    loss = model.compute_loss(x_norm, c_norm[:, j].unsqueeze(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

        print(f"Round: {round+1}\tProxy models trained.")

        
        #---------------------------------------------------------------------------
        # Flow model Training Part
        prior_model = FlowModel(x_dim=dim, hidden_dim=args.prior_hidden_dim, step_size=args.flow_steps, device=device, dtype=dtype).to(dtype=dtype, device=device)
        prior_model_optimizer = torch.optim.Adam(prior_model.parameters(), lr=1e-3)
        prior_model.train()
        for epoch in tqdm(range(num_prior_epochs), dynamic_ncols=True):
            total_loss = 0.0
            for x, y, c in data_loader:
                x = (x - test_function.X_mean) / (test_function.X_std + 1e-7)
                x += torch.randn_like(x) * 0.001
                prior_model_optimizer.zero_grad()
                loss = prior_model.compute_loss(x)
                loss.backward()
                prior_model_optimizer.step()
                total_loss += loss.item()
        print(f"Round: {round+1}\tPrior model trained")

        #---------------------------------------------------------------------------
        # Diffusion Sampler Training Part
        gfn_model = GFN(dim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                        trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                        langevin=args.langevin, learned_variance=args.learned_variance,
                        partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                        pb_scale_range=args.pb_scale_range,
                        t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                        conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                        pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                        joint_layers=args.joint_layers, zero_init=args.zero_init, device=device, dtype=dtype).to(device)

        
        gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                        args.conditional_flow_model, args.use_weight_decay, args.weight_decay)
        metrics = dict()
        energy = Energy(proxy_model_ens, prior_model, beta=args.beta, alpha = args.alpha)

        buffer = ReplayBuffer(args.gfn_buffer_size, device, energy, args.gfn_batch_size, data_ndim=dim, beta=args.beta,
                            rank_weight=args.rank_weight, prioritized=args.prioritized)
        buffer_ls = ReplayBuffer(args.gfn_buffer_size, device, energy, args.gfn_batch_size, data_ndim=dim, beta=args.beta,
                            rank_weight=args.rank_weight, prioritized=args.prioritized)
        gfn_model.train()
        diffusion_sampler = DiffusionSampler(energy, prior_model, gfn_model, buffer, buffer_ls, device, dtype, args.gfn_batch_size, args, beta=1)
        for i in trange(num_posterior_epochs+1):
            loss = diffusion_sampler.train(i)
            loss.backward()
            gfn_optimizer.step()
            print(f"Round: {round+1}\tEpoch: {i+1}\tLoss: {loss.item():.3f}")
        print(f"Round: {round+1}\tPosterior model trained")
        
        #---------------------------------------------------------------------------
        ## Sample from the diffusion sampler
        X_sample_total = []
        logR_sample_total = []
        if args.filtering == "true":
            for _ in tqdm(range(args.M)): #NOTE we sample batchsize * M * M samples total
                X_sample = diffusion_sampler.sample(batch_size * args.M, track_gradient=False)
                logr = proxy_model_ens.log_reward(X_sample)
                X_sample = X_sample.detach().clone()
                logr = logr.detach().clone()
                X_sample_total.append(X_sample)
                logR_sample_total.append(logr)
                
            X_sample = torch.cat(X_sample_total, dim=0)
            logR_sample = torch.cat(logR_sample_total, dim=0)

            #filter largest logR sample with batchsize
            X_sample = X_sample[torch.argsort(logR_sample, descending=True)[:batch_size]]
        else:
            X_sample = diffusion_sampler.sample(batch_size, track_gradient=False)
 

        #---------------------------------------------------------------------------
        ## Evaluation Part
        X_sample_unnorm = X_sample * test_function.X_std + test_function.X_mean
        X_sample_unnorm = torch.clamp(X_sample_unnorm, 0.0, 1.0)
        Y_sample_unnorm, C_sample_unnorm, true_score = test_function.eval_all(X_sample_unnorm)
        print(f"Round: {round+1}\tSeed: {seed}\tMax in this round: {Y_sample_unnorm.max().item():.3f}\tMin Constraint: {C_sample_unnorm.min().item():.3f}\t Log_rewards: {proxy_model_ens.log_reward(X_sample_unnorm).max().item():.3f}\t True score: {true_score.max().item():.3f}")
        
        test_function.X = torch.cat([test_function.X, X_sample_unnorm], dim=0)
        test_function.Y = torch.cat([test_function.Y, Y_sample_unnorm], dim=0)
        test_function.C = torch.cat([test_function.C, C_sample_unnorm], dim=0)
        test_function.true_score = torch.cat([test_function.true_score, true_score], dim=0)
        
        mean_constraint_violation = torch.mean(torch.clamp(C_sample_unnorm, min=0.0), dim=0)
        print(f"Round: {round+1}\tSeed: {seed}\tMax so far: {test_function.Y.max().item():.3f}\tMin Constraint: {test_function.C.min().item():.3f}\t Log_rewards: {proxy_model_ens.log_reward(test_function.X).max().item():.3f}\t True score: {test_function.true_score.max().item():.3f}")
        
        print(f"Time taken: {time.time() - start_time:.3f} seconds")
        print()
        
        norm_Y = (test_function.Y - test_function.Y_mean) / (test_function.Y_std + 1e-7)
        norm_C = test_function.normalizing_constraints(test_function.C)
        weights = get_lagrangian(norm_Y.squeeze(), norm_C, lamb, indicator)
        
        idx = torch.argsort(weights, descending=True)[:args.buffer_size]
        test_function.X = test_function.X[idx]
        test_function.Y = test_function.Y[idx]
        test_function.C = test_function.C[idx]
        test_function.true_score = test_function.true_score[idx]
        X_total = np.concatenate([X_total, X_sample_unnorm.cpu().numpy()], axis=0)
        Y_total = np.concatenate([Y_total, Y_sample_unnorm.cpu().numpy()], axis=0)
        C_total = np.concatenate([C_total, C_sample_unnorm.cpu().numpy()], axis=0)
        True_total = np.concatenate([True_total, true_score.cpu().numpy()], axis=0)
        
        if args.wandb == "true":
            args_list = {
                "Round": round + 1,
                "Max so far": Y_total.max().item(),
                "Max in this round": Y_sample_unnorm.max().item(),
                "Min Constraint so far": test_function.C.min().item(),
                "Min Constraint in this round": C_sample_unnorm.min().item(),
                "Max Log rewards": proxy_model_ens.log_reward(test_function.X).max().item(),
                "Max True score": True_total.max().item(),
                "Time taken": time.time() - start_time,
                "Seed": seed,
            }
            for i in range(test_function.C.shape[1]):
                args_list[f"Mean Constraint Violation {i+1}"] = mean_constraint_violation[i]            
            wandb.log(args_list)
        
        save_len = min(len(Y_total) // 1000 * 1000, args.max_evals)
        save_np = True_total[:save_len]
        file_name = f"CiBO_{task}_{dim}_{seed}_{n_init}_{args.indicator}_{args.lamb}_{args.batch_size}_{args.buffer_size}_{args.max_evals}_{save_len}.npy"
        save_numpy_array(path=args.save_path, array=save_np, file_name=file_name)
        
        previous_file_name = f"CiBO_{task}_{dim}_{seed}_{n_init}_{args.indicator}_{args.lamb}_{args.batch_size}_{args.buffer_size}_{args.max_evals}_{save_len-1000}.npy"
        if os.path.exists(os.path.join(args.save_path, previous_file_name)):
            os.remove(os.path.join(args.save_path, previous_file_name))