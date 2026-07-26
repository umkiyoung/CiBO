import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from baselines.functions.test_function import TestFunction
from baselines.utils import save_numpy_array, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sobol sequence experiment")
    parser.add_argument("-f", "--task", default="Ackley", type=str)
    parser.add_argument("--dim", default=200, type=int)
    parser.add_argument("--max_evals", default=10000, type=int)
    parser.add_argument("--n_init", default=200, type=int)
    parser.add_argument("--batch_size", default=100, type=int, help="save frequency")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--indicator", default="false", type=str)
    parser.add_argument("--save_path", default="./baselines/results/sobol/", type=str)
    args = parser.parse_args()
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

    n_remaining = max_evals - n_init
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    sobol_points = sobol.draw(n_remaining).to(dtype=dtype, device=device)

    prev_save_len = 0
    _batch_oracle_acc = 0.0
    _batch_total_start = time.time()

    for i in range(n_remaining):
        x_tensor = sobol_points[i].unsqueeze(0)

        if i % batch_size == 0:
            _batch_oracle_acc = 0.0
            _batch_total_start = time.time()

        start = time.time()
        Y_, C_, True_ = test_function.eval_all(x_tensor)
        _batch_oracle_acc += time.time() - start

        test_function.X = torch.cat((test_function.X, x_tensor), dim=0)
        test_function.Y = torch.cat((test_function.Y, Y_), dim=0)
        test_function.C = torch.cat((test_function.C, C_), dim=0)
        test_function.true_score = torch.cat((test_function.true_score, True_), dim=0)

        True_total = torch.cat((True_total, True_), dim=0)

        n_total = n_init + i + 1
        print(
            f"Eval {n_total}/{max_evals}, best true score: {True_total.max().item():.4f}, "
            f"time: {time.time() - start:.2f}s"
        )

        if (i + 1) % batch_size == 0:
            _bt = time.time() - _batch_total_start
            _round = (i + 1) // batch_size
            print(f"[TIME] Round {_round}: surrogate=0.00s generative=0.00s candidates=0.00s oracle={_batch_oracle_acc:.2f}s total={_bt:.2f}s")

        save_len = min((n_total // batch_size) * batch_size, max_evals)
        if save_len > prev_save_len:
            save_np = True_total[:save_len].cpu().numpy()
            file_name = (
                f"sobol_{task}_{dim}_{seed}_{n_init}_{indicator}_"
                f"{batch_size}_{max_evals}_{save_len}.npy"
            )
            prev_file_name = (
                f"sobol_{task}_{dim}_{seed}_{n_init}_{indicator}_"
                f"{batch_size}_{max_evals}_{prev_save_len}.npy"
            )
            save_numpy_array(args.save_path, save_np, file_name)
            prev_path = os.path.join(args.save_path, prev_file_name)
            if prev_save_len > 0 and os.path.exists(prev_path):
                os.remove(prev_path)
            prev_save_len = save_len

    # Final save
    save_len = min(len(True_total), max_evals)
    save_np = True_total[:save_len].cpu().numpy()
    file_name = (
        f"sobol_{task}_{dim}_{seed}_{n_init}_{indicator}_"
        f"{batch_size}_{max_evals}_{save_len}.npy"
    )
    save_numpy_array(args.save_path, save_np, file_name)
