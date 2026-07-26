import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from scipy.optimize import minimize

from baselines.functions.test_function import TestFunction
from baselines.utils import save_numpy_array, set_seed, get_lagrangian


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COBYLA experiment")
    parser.add_argument("-f", "--task", default="Ackley", type=str)
    parser.add_argument("--dim", default=200, type=int)
    parser.add_argument("--max_evals", default=10000, type=int)
    parser.add_argument("--n_init", default=200, type=int)
    parser.add_argument("--batch_size", default=100, type=int, help="save frequency")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lamb", default=1.0, type=float)
    parser.add_argument("--indicator", default="false", type=str)
    parser.add_argument("--save_path", default="./baselines/results/cobyla/", type=str)
    args = parser.parse_args()
    args.buffer_size = args.max_evals
    set_seed(args.seed)

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = args.task
    dim = args.dim
    max_evals = args.max_evals
    n_init = args.n_init
    batch_size = args.batch_size
    seed = args.seed
    indicator = args.indicator

    test_function = TestFunction(
        task=task,
        dim=dim,
        n_init=n_init,
        seed=seed,
        indicator=indicator,
        dtype=dtype,
        device=device,
    )

    train_X, train_Y, train_C = test_function.get_initial_points()
    True_total = test_function.true_score.clone()

    # Start from the best initial point by Lagrangian value
    lag_init = get_lagrangian(train_Y.squeeze(), train_C, args.lamb, args.indicator)
    best_idx = int((-lag_init).argmin())
    x0 = train_X[best_idx].cpu().numpy()

    # ------------------------------------------------------------------ #
    # Evaluation callback                                                  #
    # ------------------------------------------------------------------ #
    true_total_ref = [True_total]   # mutable holder so the closure can update it
    eval_count = [0]
    prev_save_len = [0]
    _batch_oracle_acc = [0.0]
    _batch_total_start = [None]

    def objective(x):
        x_np = np.clip(x, 0.0, 1.0)
        x_tensor = torch.tensor(x_np, dtype=dtype, device=device).unsqueeze(0)

        batch_idx = eval_count[0] % batch_size
        if batch_idx == 0:
            _batch_oracle_acc[0] = 0.0
            _batch_total_start[0] = time.time()

        start = time.time()
        Y_, C_, True_ = test_function.eval_all(x_tensor)
        _batch_oracle_acc[0] += time.time() - start

        # Keep test_function internal state consistent (mirrors CMA-ES pattern)
        test_function.X = torch.cat((test_function.X, x_tensor), dim=0)
        test_function.Y = torch.cat((test_function.Y, Y_), dim=0)
        test_function.C = torch.cat((test_function.C, C_), dim=0)
        test_function.true_score = torch.cat((test_function.true_score, True_), dim=0)

        true_total_ref[0] = torch.cat((true_total_ref[0], True_), dim=0)
        eval_count[0] += 1

        n_total = n_init + eval_count[0]
        print(
            f"Eval {n_total}/{max_evals}, best true score: {true_total_ref[0].max().item():.4f}, "
            f"time: {time.time() - start:.2f}s"
        )

        if (eval_count[0] % batch_size == 0) and _batch_total_start[0] is not None:
            _bt = time.time() - _batch_total_start[0]
            _bo = _batch_oracle_acc[0]
            _round = eval_count[0] // batch_size
            print(f"[TIME] Round {_round}: surrogate=0.00s generative=0.00s candidates={_bt - _bo:.2f}s oracle={_bo:.2f}s total={_bt:.2f}s")

        # Save every batch_size evaluations
        save_len = min((n_total // batch_size) * batch_size, max_evals)
        if save_len > prev_save_len[0]:
            save_np = true_total_ref[0][:save_len].cpu().numpy()
            file_name = (
                f"cobyla_{task}_{dim}_{seed}_{n_init}_{indicator}_"
                f"{args.lamb}_{batch_size}_{args.buffer_size}_{max_evals}_{save_len}.npy"
            )
            prev_file_name = (
                f"cobyla_{task}_{dim}_{seed}_{n_init}_{indicator}_"
                f"{args.lamb}_{batch_size}_{args.buffer_size}_{max_evals}_{prev_save_len[0]}.npy"
            )
            save_numpy_array(args.save_path, save_np, file_name)
            prev_path = os.path.join(args.save_path, prev_file_name)
            if prev_save_len[0] > 0 and os.path.exists(prev_path):
                os.remove(prev_path)
            prev_save_len[0] = save_len

        lag = get_lagrangian(Y_.squeeze(), C_, args.lamb, args.indicator)
        return -lag.item()  # scipy minimizes; negate to maximize Lagrangian

    # ------------------------------------------------------------------ #
    # Run COBYLA                                                           #
    # ------------------------------------------------------------------ #
    set_seed(args.seed)
    n_remaining = max_evals - n_init

    minimize(
        objective,
        x0,
        method="COBYLA",
        options={
            "maxiter": n_remaining,
            "rhobeg": 0.1,   # initial trust region radius on [0,1]^d
            "catol": 0.0,
        },
    )

    # Final save
    save_len = min(len(true_total_ref[0]), max_evals)
    save_np = true_total_ref[0][:save_len].cpu().numpy()
    file_name = (
        f"cobyla_{task}_{dim}_{seed}_{n_init}_{indicator}_"
        f"{args.lamb}_{batch_size}_{args.buffer_size}_{max_evals}_{save_len}.npy"
    )
    save_numpy_array(args.save_path, save_np, file_name)
