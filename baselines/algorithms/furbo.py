import math
import os
import sys
import warnings
import gc
from dataclasses import dataclass, field

# Ensure the project root (Constrained_PIBO/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
import argparse

import time
from baselines.functions.test_function import TestFunction
from baselines.utils import set_seed, save_numpy_array
import numpy as np
import wandb

def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FuRBO experiment")
    parser.add_argument("-f", "--task", default="Ackley", type=str)
    parser.add_argument("--dim", default=200, type=int)
    parser.add_argument("--max_evals", default=10000, type=int)
    parser.add_argument("--n_init", default=200, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--indicator", default="false", type=str)
    parser.add_argument("--inspector_percentage", default=0.1, type=float,
                        help="fraction of inspector points that define the trust region")
    parser.add_argument("--wandb", default="false", type=str)
    parser.add_argument("--save_path", default="./baselines/results/furbo/", type=str)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    use_wandb = args.wandb.lower() == "true"
    if use_wandb:
        wandb.init(
            project="furbo",
            name=f"{args.task}_{args.dim}_{args.seed}_{args.n_init}_{args.indicator}_{args.batch_size}_{args.max_evals}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    max_cholesky_size = 1024

    task = args.task
    dim = args.dim
    max_evals = args.max_evals
    n_init = args.n_init
    batch_size = args.batch_size
    seed = args.seed
    indicator = args.indicator
    inspector_percentage = args.inspector_percentage

    set_seed(seed)

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # ------------------------------------------------------------------ #
    # State                                                                #
    # ------------------------------------------------------------------ #
    @dataclass
    class FurboState:
        dim: int
        batch_size: int
        radius: float = 1.0
        radius_min: float = 0.5 ** 7          # restart threshold
        success_counter: int = 0
        success_tolerance: int = 2            # τ_s from FuRBO paper
        failure_counter: int = 0
        failure_tolerance: int = 3            # τ_f from FuRBO paper
        best_value: float = -float("inf")
        best_constraint_values: Tensor = field(
            default_factory=lambda: torch.ones(2, device=device, dtype=dtype) * torch.inf
        )
        restart_triggered: bool = False

    def update_tr_radius(state: FurboState):
        if state.success_counter == state.success_tolerance:
            state.radius = min(2.0 * state.radius, 1.0)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:
            state.radius /= 2.0
            state.failure_counter = 0

        if state.radius < state.radius_min:
            state.restart_triggered = True

        return state

    # ------------------------------------------------------------------ #
    # Helpers shared with SCBO                                            #
    # ------------------------------------------------------------------ #
    def get_best_index_for_batch(Y: Tensor, C: Tensor):
        is_feas = (C <= 0).all(dim=-1)
        if is_feas.any():
            score = Y.clone()
            score[~is_feas] = -float("inf")
            return score.argmax()
        return C.clamp(min=0).sum(dim=-1).argmin()

    def update_state(state, Y_next, C_next):
        best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
        y_next, c_next = Y_next[best_ind], C_next[best_ind]

        if (c_next <= 0).all():
            improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
            if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1
        else:
            total_violation_next = c_next.clamp(min=0).sum(dim=-1)
            total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
            if total_violation_next < total_violation_center:
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = y_next.item()
                state.best_constraint_values = c_next
            else:
                state.success_counter = 0
                state.failure_counter += 1

        state = update_tr_radius(state)
        return state

    # ------------------------------------------------------------------ #
    # FuRBO: inspector-based trust region construction                    #
    # ------------------------------------------------------------------ #
    def sample_hypersphere(center: Tensor, radius: float, n_samples: int) -> Tensor:
        """Uniform samples inside a hypersphere of given radius centred at `center`."""
        d = center.shape[0]
        z = torch.randn(n_samples, d, **tkwargs)
        z = z / z.norm(dim=1, keepdim=True)
        r = torch.rand(n_samples, 1, **tkwargs) * radius
        samples = center + z * r
        return torch.clamp(samples, 0.0, 1.0)

    def compute_tr_bounds(
        state: FurboState,
        model,
        constraint_model,
        X: Tensor,
        Y: Tensor,
        C: Tensor,
        n_inspector: int,
        percentage: float = 0.1,
    ):
        """FuRBO's feasibility-aware inspector sampling to build TR bounds."""
        x_center = X[get_best_index_for_batch(Y, C)].clone()

        samples = sample_hypersphere(x_center, state.radius, n_inspector)

        # Evaluate GP posterior means on inspector points
        model.eval()
        constraint_model.eval()
        with torch.no_grad():
            yy = model.posterior(samples).mean.squeeze(-1)           # [n]
            cc = constraint_model.posterior(samples).mean            # [n, K]

        # Normalize each constraint then take per-sample max violation
        scale = cc.abs().max(dim=0).values.clamp(min=1e-8)
        cc_norm = cc / scale
        cc_max = cc_norm.max(dim=1).values                           # [n]

        n_tr = max(4, int(n_inspector * percentage))

        if (cc_max < 0).any():
            feas_idx = torch.where(cc_max <= 0)[0]
            infeas_idx = torch.where(cc_max > 0)[0]
            # Feasible: sorted by best objective (descending)
            feas_sorted = feas_idx[torch.argsort(-yy[feas_idx])]
            # Infeasible: sorted by smallest violation (ascending)
            infeas_sorted = infeas_idx[torch.argsort(cc_max[infeas_idx])]
            top_idx = torch.cat([feas_sorted, infeas_sorted])[:n_tr]
        else:
            top_idx = torch.argsort(cc_max)[:n_tr]

        tr_lb = samples[top_idx].min(dim=0).values
        tr_ub = samples[top_idx].max(dim=0).values
        return tr_lb, tr_ub

    # ------------------------------------------------------------------ #
    # Candidate generation (Thompson sampling within TR)                  #
    # ------------------------------------------------------------------ #
    def generate_batch(
        state: FurboState,
        model,
        X: Tensor,
        Y: Tensor,
        C: Tensor,
        batch_size: int,
        n_candidates: int,
        constraint_model,
        sobol: SobolEngine,
        tr_lb: Tensor,
        tr_ub: Tensor,
    ) -> Tensor:
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        best_ind = get_best_index_for_batch(Y=Y, C=C)
        x_center = X[best_ind, :].clone()

        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
        )
        with torch.no_grad():
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        del pert, mask, X_cand
        clear_gpu_memory()
        return X_next

    # ------------------------------------------------------------------ #
    # GP fitting (identical to SCBO)                                      #
    # ------------------------------------------------------------------ #
    def get_fitted_model(X, Y):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(
            X, Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"[Warning] GP fitting failed: {e!r}")
            state.restart_triggered = True
        return model

    # ------------------------------------------------------------------ #
    # Initialise                                                           #
    # ------------------------------------------------------------------ #
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
    num_constraints = train_C.shape[1]
    C_list = [train_C[:, i].unsqueeze(-1) for i in range(num_constraints)]

    X_total = train_X.clone()
    Y_total = train_Y.clone()
    C_total = train_C.clone()
    True_total = test_function.true_score

    state = FurboState(dim=dim, batch_size=batch_size)
    # Mopta has 68 constraints; evaluating 5000 candidates across that many GPs
    # causes GPU OOM. Use a lower cap for high-constraint tasks.
    if task.lower() == "mopta":
        N_CANDIDATES = min(1000, max(200, 10 * dim))
    else:
        N_CANDIDATES = min(5000, max(2000, 200 * dim))
    sobol = SobolEngine(dim, scramble=True)

    # ------------------------------------------------------------------ #
    # Main optimisation loop                                               #
    # ------------------------------------------------------------------ #
    iteration = 0
    while len(X_total) < max_evals:
        iteration += 1
        print(f"\n===== Iteration {iteration} =====")
        clear_gpu_memory()

        if state.restart_triggered:
            state = FurboState(dim=dim, batch_size=batch_size)
            train_X, train_Y, train_C = test_function.get_initial_points()
            C_list = [train_C[:, i].unsqueeze(-1) for i in range(num_constraints)]
            X_total = torch.cat((X_total, train_X), dim=0)
            Y_total = torch.cat((Y_total, train_Y), dim=0)
            C_total = torch.cat((C_total, train_C), dim=0)
            true_score = test_function.true_score
            True_total = torch.cat((True_total, true_score), dim=0)
            print(f"Restart! Max so far: {True_total.max():.2e}")
            clear_gpu_memory()

        _t_surrogate = _sync_time()
        print("Fitting objective model...")
        model = get_fitted_model(train_X, train_Y)

        print("Fitting constraint models...")
        const_model_list = [get_fitted_model(train_X, C_list[i]) for i in range(num_constraints)]
        _time_surrogate = _sync_time() - _t_surrogate
        clear_gpu_memory()

        _t_candidates = _sync_time()
        C_cat = torch.cat(C_list, dim=-1)
        const_model_list_gp = ModelListGP(*const_model_list)

        print("Computing trust region bounds...")
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            tr_lb, tr_ub = compute_tr_bounds(
                state=state,
                model=model,
                constraint_model=const_model_list_gp,
                X=train_X,
                Y=train_Y,
                C=C_cat,
                n_inspector=N_CANDIDATES,
                percentage=inspector_percentage,
            )

            print("Generating candidate batch...")
            X_next = generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=C_cat,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                constraint_model=const_model_list_gp,
                sobol=sobol,
                tr_lb=tr_lb,
                tr_ub=tr_ub,
            )

        _time_candidates = _sync_time() - _t_candidates
        del model, const_model_list, const_model_list_gp
        clear_gpu_memory()

        _t_oracle = _sync_time()
        Y_next, C_next, true_score = test_function.eval_all(X_next)
        _time_oracle = _sync_time() - _t_oracle

        state = update_state(state=state, Y_next=Y_next, C_next=C_next)

        train_X = torch.cat((train_X, X_next), dim=0)
        train_Y = torch.cat((train_Y, Y_next), dim=0)
        for i in range(num_constraints):
            C_list[i] = torch.cat((C_list[i], C_next[:, i].unsqueeze(-1)), dim=0)

        X_total = torch.cat((X_total, X_next), dim=0)
        Y_total = torch.cat((Y_total, Y_next), dim=0)
        C_total = torch.cat((C_total, C_next), dim=0)
        True_total = torch.cat((True_total, true_score), dim=0)

        print(f"Current Max Score: {true_score.max():.2e}, Max so far: {True_total.max():.2e}")

        if (state.best_constraint_values <= 0).all():
            print(f"{len(X_total)}) Best value: {state.best_value:.2e}, radius: {state.radius:.2e}")
        else:
            violation = state.best_constraint_values.clamp(min=0).sum()
            print(
                f"{len(X_total)}) No feasible point yet! Smallest total violation: "
                f"{violation:.2e}, radius: {state.radius:.2e}"
            )

        _time_total = _time_surrogate + _time_candidates + _time_oracle
        print(f"[TIME] Round {iteration}: surrogate={_time_surrogate:.2f}s generative=0.00s candidates={_time_candidates:.2f}s oracle={_time_oracle:.2f}s total={_time_total:.2f}s")

        if use_wandb:
            wandb.log({
                "Round": iteration,
                "Max so far": Y_total.max().item(),
                "Max in this round": Y_next.max().item(),
                "Min Constraint so far": C_total.min().item(),
                "Min Constraint in this round": C_next.min().item(),
                "Max True score": True_total.max().item(),
                "radius": state.radius,
            })

        save_len = min(len(True_total), max_evals)
        save_np = True_total[:save_len].cpu().numpy()
        file_name = (
            f"furbo_{task}_{dim}_{seed}_{n_init}_{indicator}_"
            f"{batch_size}_{max_evals}_{len(True_total)}.npy"
        )
        previous_file_name = (
            f"furbo_{task}_{dim}_{seed}_{n_init}_{indicator}_"
            f"{batch_size}_{max_evals}_{len(True_total) - batch_size}.npy"
        )
        save_numpy_array(path=args.save_path, array=save_np, file_name=file_name)

        prev_path = os.path.join(args.save_path, previous_file_name)
        if os.path.exists(prev_path):
            os.remove(prev_path)

        clear_gpu_memory()
