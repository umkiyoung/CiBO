import argparse
import torch
import numpy as np
from cma import CMAEvolutionStrategy, ConstrainedFitnessAL
import argparse
from baselines.utils import save_numpy_array, set_seed, get_lagrangian
from baselines.functions.test_function import TestFunction
import time
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BO experiment')
    parser.add_argument('-f', '--task', default='Ackley', type=str, help='function to optimize')
    parser.add_argument('--dim', default=200, type=int, help='dimension of the function')
    parser.add_argument('--max_evals', default=10000, type=int, help='evaluation number')
    parser.add_argument('--n_init', default=200, type=int, help=' number of initial points')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size of each iteration')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lamb', default=1.0, type=float, help='Lagrangian multiplier')
    parser.add_argument('--indicator', default='false', type=str, help='indicator function')
    parser.add_argument('--save_path', default='./baselines/results/cmaes/', type=str, help='path to save the results')
    args = parser.parse_args()    
    args.buffer_size = args.max_evals
    set_seed(args.seed)
    
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, dim, max_evals, n_init, batch_size, seed, indicator = args.task, args.dim, args.max_evals, args.n_init, args.batch_size, args.seed, args.indicator
    test_function = TestFunction(
        task=task,
        dim=dim,
        n_init=n_init, 
        seed=seed,
        indicator=indicator,
        dtype=dtype,
        device=device,
    )
    train_X , train_Y , train_C = test_function.get_initial_points()
    True_total = test_function.true_score
    X_total, Y_total, C_total = train_X.clone(), train_Y.clone(), train_C.clone()
    c_dim = C_total.shape[1]
    
    fun = test_function.eval_objective
    
    
    dim = args.dim
    
    
    # Use lagrangian as a Y
    lag_val = get_lagrangian(train_Y.squeeze(), train_C, args.lamb, args.indicator)
    Y0 = lag_val.cpu().numpy()
    X0 = train_X.cpu().numpy()
    
    Y0 = Y0 * -1
    current_best = np.min(Y0)
    best_idx = np.argmin(Y0)
    x0 = X0[best_idx]
    set_seed(args.seed)
    options = {
        'bounds': [0, 1],
        'popsize': args.batch_size,
        'verb_filenameprefix': 'outcmaes/',
    }
    
    #optimization
    sigma0 = 0.1
    es = CMAEvolutionStrategy(x0, sigma0, options)
    n_generations = (args.max_evals-args.n_init)//args.batch_size
    #Optimization Loop
    for generation in range(n_generations):
        start_time = time.time()
        _t_cand = time.time()
        X = es.ask()
        _time_candidates = time.time() - _t_cand

        X_  = np.array(X)
        X_= torch.tensor(X_, dtype=dtype, device=device)
        _t_oracle = time.time()
        Y_, C_, True_ = test_function.eval_all(X_)
        _time_oracle = time.time() - _t_oracle
        
        test_function.X = torch.cat((test_function.X, X_), dim=0)
        test_function.Y = torch.cat((test_function.Y, Y_), dim=0)
        test_function.C = torch.cat((test_function.C, C_), dim=0)
        test_function.true_score = torch.cat((test_function.true_score, True_), dim=0)
        
        True_total = torch.cat((True_total, True_), dim=0)
        
        Y_ = get_lagrangian(Y_.squeeze(), C_, args.lamb, args.indicator)
        
        
        Y = -1 * Y_
        Y = Y.cpu().numpy().tolist()
        _t_tell = time.time()
        es.tell(X, Y)
        _time_surrogate = time.time() - _t_tell
        end_time = time.time()
        _time_total = _time_surrogate + _time_candidates + _time_oracle
        print(f'Generation {generation+1}/{n_generations}, best value: {-np.min(Y)}, time(min): {(end_time-start_time)/60}, best true score: {True_total.max().item()}')
        print(f"[TIME] Round {generation+1}: surrogate={_time_surrogate:.2f}s generative=0.00s candidates={_time_candidates:.2f}s oracle={_time_oracle:.2f}s total={_time_total:.2f}s")
        
        if es.stop():
            break
        
        save_len = min(len(True_total) // 1000 * 1000, args.max_evals)
        save_np = True_total[:save_len].cpu().numpy()
        file_name = f"cmaes_{task}_{dim}_{seed}_{args.n_init}_{args.indicator}_{args.lamb}_{args.batch_size}_{args.buffer_size}_{args.max_evals}_{save_len}.npy"
        save_numpy_array(args.save_path, save_np, file_name)
        previous_file_name = f"cmaes_{task}_{dim}_{seed}_{args.n_init}_{args.indicator}_{args.lamb}_{args.batch_size}_{args.buffer_size}_{args.max_evals}_{save_len-1000}.npy"
        if os.path.exists(os.path.join(args.save_path, previous_file_name)):
            os.remove(os.path.join(args.save_path, previous_file_name))            
    