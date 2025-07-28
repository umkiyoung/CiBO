from typing import Tuple

from botorch.test_functions import Ackley, Rastrigin, Rosenbrock
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval
from concurrent.futures import ThreadPoolExecutor
from torch import Tensor
import numpy as np
import time
from functions.mopta08 import MoptaSoftConstraints, evaluate_batch_parallel
import torch
from torch.utils.data import Dataset
from torch.quasirandom import SobolEngine
from functions.rover_planning import Rover
from functions.lasso_benchmark import LassoDNABenchmark



class TestFunction(Dataset):
    def __init__(self, task: str, dim: int = 200, n_init: int = 200, seed: int = 0, cons=30, indicator='false', dtype=torch.float64, device='cpu', negate=True,):
        self.task = task
        self.dim = dim
        self.n_init = n_init
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.indicator = indicator
        self.lb, self.ub = None, None
        self.constraints = []
        self.constraints_coeff = cons
        #NOTE: Synthetic Functions
        if task == 'Ackley':
            # Constrain Settings:
            # c1(x) = ∑10  i=1 xi ≤ 0 and c2(x) = ‖x‖2 − 5 ≤ 0. (SCBO)
            self.fun = Ackley(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 10 #Following TurBO
            
            def c1(x):
                return torch.sum(x, dim=-1) - 0
            def c2(x):
                return torch.norm(x, p=2, dim=-1) - self.constraints_coeff # 30 #NOTE: Adjusted to 30 for 200D

            def eval_c1(x):
                return c1(unnormalize(x, self.fun.bounds))
            def eval_c2(x):
                return c2(unnormalize(x, self.fun.bounds))
            
            self.constraints.append((c1, eval_c1))
            self.constraints.append((c2, eval_c2))

        

        elif task == 'Rastrigin':
            self.fun = Rastrigin(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 5 #Following MCMC_BO

            def c1(x):
                return torch.sum(x, dim=-1) - 0
            def c2(x):
                return torch.norm(x, p=2, dim=-1) - self.constraints_coeff #NOTE: Adjusted to 30 for 200D
                        
            def eval_c1(x):
                return c1(unnormalize(x, self.fun.bounds))
            def eval_c2(x):
                return c2(unnormalize(x, self.fun.bounds))
            
            self.constraints.append((c1, eval_c1))
            self.constraints.append((c2, eval_c2))
            
        elif task == 'Rosenbrock':
            self.fun = Rosenbrock(dim=dim, negate = negate).to(dtype=dtype, device=device)
            self.lb, self.ub = -5, 10 #Following LA-MCTS

            def c1(x):
                return torch.sum(x, dim=-1) - 0
            def c2(x):
                return torch.norm(x, p=2, dim=-1) - 30 #NOTE: Adjusted to 30 for 200D
            
            def eval_c1(x):
                return c1(unnormalize(x, self.fun.bounds))
            def eval_c2(x):
                return c2(unnormalize(x, self.fun.bounds))
            
            self.constraints.append((c1, eval_c1))
            self.constraints.append((c2, eval_c2))
        
        elif task == 'DNA':
            self.fun = LassoDNABenchmark(seed=seed, dtype=dtype, device=device, constraints_coeff=cons)  
            self.lb, self.ub = -1, 1
            
        elif task == 'Mopta':
            self.fun = MoptaSoftConstraints(self.dim, self.dtype, self.device)
            self.lb, self.ub = 0, 1           
            
        elif task == 'RoverPlanning':
            self.fun = Rover(dim=dim, dtype=dtype, device=device, force_initial_end=True)
            self.lb, self.ub = 0, 1           
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        if self.lb is not None and self.ub is not None:
            self.fun.bounds[0, :].fill_(self.lb)
            self.fun.bounds[1, :].fill_(self.ub)
            self.fun.bounds.to(dtype=dtype, device=device)

    def feasible_sampler(self, n_samples, dim, burn_in=500, thin=10, R=30.0):
        # 1) Initialize
        x = torch.full((dim,), -1.0, device=self.device, dtype=self.dtype)
        l = torch.full((dim,), self.lb, device=self.device, dtype=self.dtype)
        u = torch.full((dim,), self.ub, device=self.device, dtype=self.dtype)
        samples = []
        total_iters = burn_in + n_samples * thin

        for it in range(total_iters):
            # 2) Random unit direction
            d = torch.randn(dim, device=self.device, dtype=self.dtype)
            d /= d.norm()

            # 3-a) Box constraints -> compute t_min and t_max vectorized
            t_low  = torch.where(d > 0, (l - x) / d, (u - x) / d)
            t_high = torch.where(d > 0, (u - x) / d, (l - x) / d)
            t_min, t_max = t_low.max().item(), t_high.min().item()

            # 3-b) Half-space sum(x + t d) <= 0
            s_d, s_x = d.sum().item(), x.sum().item()
            if abs(s_d) > 1e-12:
                bound = ((dim - s_x) if self.task == 'Levy' else -s_x) / s_d
                if s_d > 0:
                    t_max = min(t_max, bound)
                else:
                    t_min = max(t_min, bound)

            # 3-c) Ball constraint ||x + t d||^2 <= R^2
            y = x.clone() - (1.0 if self.task == 'Levy' else 0.0)   # shift center to 1
            b  = 2.0 * y.dot(d)
            c  = y.dot(y) - R**2
            disc = b * b - 4.0 * c
            sqrt_disc = torch.sqrt(disc)
            t1, t2 = (-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0
            t_min = max(t_min, t1.item())
            t_max = min(t_max, t2.item())

            # 4) Uniform step and update
            t = torch.empty(1, device=self.device, dtype=self.dtype).uniform_(t_min, t_max).item()
            x += t * d

            # 5) Collect after burn-in, thinning
            if it >= burn_in and (it - burn_in) % thin == 0:
                samples.append(x.clone())
        unnormalize_X = torch.stack(samples, dim=0)
        normalize_X = (unnormalize_X - self.lb) / (self.ub - self.lb)
        return normalize_X
      
    def indicator_function(self, constraint):
        """
        if value of the tensor > 0 return 1 else 0
        """
        if self.indicator == 'true':
            return torch.where(constraint > 0, torch.ones_like(constraint), torch.zeros_like(constraint))
        else:
            return constraint
    
    def eval_objective(self, x):
        if self.task in ['RoverPlanning', 'DNA']:
            return self.fun(unnormalize(x, self.fun.bounds))[0]
        elif self.task in ['Mopta']:
            return evaluate_batch_parallel(x, self.fun, self.dtype, self.device,self.constraints_coeff)[0]
        else:
            return self.fun(unnormalize(x, self.fun.bounds))
    
    def eval_constraints(self, x):
        if self.task in ['RoverPlanning', 'DNA']:
            return self.indicator_function(self.fun(unnormalize(x, self.fun.bounds))[1])
        elif self.task in ['Mopta']:
            return self.indicator_function(evaluate_batch_parallel(x,self.fun,self.dtype,self.device,self.constraints_coeff)[1])   
        else:
            c_list = []
            for c, eval_c in self.constraints:
                c_list.append(eval_c(x))
            c_list = torch.stack(c_list, dim=-1)
            c_list = c_list.unsqueeze(0) if c_list.ndim == 1 else c_list

            return self.indicator_function(c_list)
    
    def eval_objective_with_constraints(self, x):
        if self.task in ['RoverPlanning', 'DNA']:
            vals, constraints = self.fun(unnormalize(x, self.fun.bounds))
            return vals, self.indicator_function(constraints)
        y = self.eval_objective(x)
        c_list = self.eval_constraints(x)

        return y, c_list
    
    def eval_score(self, x):
        if self.task in ['Mopta']: 
            # y, c_list = self.fun(unnormalize(x, self.fun.bounds))
            y, c_list = evaluate_batch_parallel(x, self.fun, self.dtype, self.device, self.constraints_coeff)
            mask = (c_list > 0).any(dim=1)
            new_c_list = torch.where(mask, float('-inf'), y).to(dtype=self.dtype, device=self.device)
            return new_c_list.unsqueeze(-1)

        y, c_list = self.eval_objective_with_constraints(x)

        # if any constraint is violated, return a -inf
        if torch.any(c_list > 0):
            return -float('inf')
        else:
            return y.item()
        
    def eval_all(self, x_batch):
        if self.task in ['Mopta']:
            y, c_list = self.eval_objective_with_constraints(x_batch)
            mask = (c_list > 0).any(dim=1)
            score = torch.where(mask, float('-inf'), y).to(dtype=self.dtype, device=self.device)
            y = y.unsqueeze(-1)
        else:
            y = torch.tensor([self.eval_objective(x) for x in x_batch], dtype=self.dtype, device=self.device).unsqueeze(-1)
            c_list = torch.cat([self.eval_constraints(x) for x in x_batch], dim=0).to(self.dtype).to(self.device)
            score = torch.tensor([self.eval_score(x) for x in x_batch], dtype=self.dtype, device=self.device).unsqueeze(-1)
        return y, c_list, score
        
    def get_initial_points(self):
        sobol = SobolEngine(self.dim, scramble=True, seed=self.seed)
        if self.indicator == 'true' and self.task in ['Ackley', 'Rastrigin', 'Rosenbrock']:
            self.X = sobol.draw(n=self.n_init-10).to(self.dtype).to(self.device)
            self.X = torch.cat([self.X, self.feasible_sampler(n_samples=10, dim=self.dim)])
        else:
            self.X = sobol.draw(n=self.n_init).to(self.dtype).to(self.device)
        self.Y, self.C, self.true_score = self.eval_all(self.X)
        return self.X, self.Y, self.C
    
    def indicator_function(self, constraint):
        """
        if value of the tensor > 0 return 1 else 0
        """
        if self.indicator == 'true':
            return torch.where(constraint > 0, torch.ones_like(constraint), torch.zeros_like(constraint))
        else:
            return constraint
    
    def normalizing_constraints(self, constraints):
        max_vals = constraints.abs().max(dim=0, keepdim=True).values
        scaled_constraints = constraints / (max_vals + 1e-08)        
        return scaled_constraints
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.C[idx]


if __name__ == "__main__":
    # test_function = TestFunction(task='DNA', dim=180, n_init=10, seed=0, indicator='false', dtype=torch.float64, device='cpu', negate=True)
    # test_function = TestFunction(task='Rastrigin', dim=200, n_init=10, seed=0, cons=30, indicator='false', dtype=torch.float64, device='cpu', negate=True)
    test_function = TestFunction(task='Mopta', dim=124, n_init=10, seed=0, cons=68, indicator='false', dtype=torch.float64, device='cpu', negate=True)
    test_function.get_initial_points()
    test_function.get_initial_points()
    print(test_function.X)
    print(test_function.X.shape)
    print(test_function.Y.shape)
    print(test_function.C.shape)
    print(test_function.true_score.shape)
    
    