import math
import os
import warnings
import gc  # 가비지 컬렉션 추가
from dataclasses import dataclass

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
from torch_pca import PCA

def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BO experiment')
    parser.add_argument('-f', '--task', default='Ackley', type=str, help='function to optimize')
    parser.add_argument('--dim', default=200, type=int, help='dimension of the function')
    parser.add_argument('--max_evals', default=10000, type=int, help='evaluation number')
    parser.add_argument('--n_init', default=200, type=int, help=' number of initial points')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size of each iteration')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--indicator', default='false', type=str, help='indicator function')
    parser.add_argument('--save_path', default='./baselines/results/pcagp_scbo/', type=str, help='path to save the results')
    parser.add_argument('--num_components', default=2, type=int, help='number of PCA components')
    parser.add_argument('--wandb', default='false', type=str, help='log to Weights & Biases')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    max_cholesky_size = float("inf") 
    task, dim, max_evals, n_init, batch_size, seed, indicator = args.task, args.dim, args.max_evals, args.n_init, args.batch_size, args.seed , args.indicator
    set_seed(seed)
    
    num_components = args.num_components
    if args.wandb.lower() == "true":
        wandb.init(project="pcagp_scbo", config=vars(args))
    else:
        wandb.init(mode="disabled")
    
    # GPU 메모리 정리 함수 추가
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @dataclass
    class ScboState:
        dim: int
        batch_size: int
        length: float = 0.8
        length_min: float = 0.5**7
        length_max: float = 1.6
        failure_counter: int = 0
        failure_tolerance: int = float("nan")  # Note: Post-initialized
        success_counter: int = 0
        success_tolerance: int = 3  # Note: The original paper uses 3
        best_value: float = -float("inf")
        best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
        restart_triggered: bool = False

        def __post_init__(self):
            self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


    def update_tr_length(state: ScboState):
        # Update the length of the trust region according to
        # success and failure counters
        # (Just as in original TuRBO paper)
        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        if state.length < state.length_min:  # Restart when trust region becomes too small
            state.restart_triggered = True

        return state


    def get_best_index_for_batch(Y: Tensor, C: Tensor):
        """Return the index for the best point."""
        is_feas = (C <= 0).all(dim=-1)
        if is_feas.any():  # Choose best feasible candidate
            score = Y.clone()
            score[~is_feas] = -float("inf")
            return score.argmax()
        return C.clamp(min=0).sum(dim=-1).argmin()


    def update_state(state, Y_next, C_next):
        """Method used to update the TuRBO state after each step of optimization."""
        # Pick the best point from the batch
        best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
        y_next, c_next = Y_next[best_ind], C_next[best_ind]

        if (c_next <= 0).all():
            # At least one new candidate is feasible
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
            # No new candidate is feasible
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

        # Update the length of the trust region according to the success and failure counters
        state = update_tr_length(state)
        return state

    def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        C,  # Constraint values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
        sobol: SobolEngine,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        # Create the TR bounds
        best_ind = get_best_index_for_batch(Y=Y, C=C)
        x_center = X[best_ind, :].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO)
        dim = X.shape[-1]
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
        )
        with torch.no_grad():
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        # 후보 생성 후 중간 변수 정리
        del pert, mask, X_cand
        clear_gpu_memory()
        
        return X_next

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
    C_list = []
    # calculate train_C dim
    num_constraints = train_C.shape[1]
    for i in range(num_constraints):
        c = train_C[:, i].unsqueeze(-1)
        C_list.append(c)
    
    
    
    total_X, total_Y, total_C = train_X.clone(), train_Y.clone(), train_C.clone()
    scores = test_function.true_score
    state = ScboState(dim, batch_size=batch_size)

    # 후보 수 조정 가능 (필요시)
    # 지금은 원래 설정을 유지합니다
    N_CANDIDATES = min(5000, max(2000, 200 * dim))
    sobol = SobolEngine(dim, scramble=True)
    
    def get_fitted_model(X, Y):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_mll(mll)

        return model
    
    iteration = 0
    while len(total_X) < max_evals:  # Run until TuRBO converges
        iteration += 1
        print(f"\n===== Iteration {iteration} =====")
        
        # 메모리 정리
        clear_gpu_memory()
        
        # When restart is triggered, we need to reinitialize the state
        if state.restart_triggered:
            state = ScboState(dim, batch_size=batch_size)
            train_X, train_Y, train_C = test_function.get_initial_points()
            C_list = []
            # calculate train_C dim
            num_constraints = train_C.shape[1]
            for i in range(num_constraints):
                c = train_C[:, i].unsqueeze(-1)
                C_list.append(c)
            total_X = torch.cat((total_X, train_X), dim=0)
            total_Y = torch.cat((total_Y, train_Y), dim=0)
            total_C = torch.cat((total_C, train_C), dim=0)
            score = test_function.true_score
            scores = torch.cat((scores, score), dim=0)
            print(f"Restart triggered! Current Max Score: {score.max():.2e}, Max so far: {scores.max():.2e}")
            
            # 재시작 후 메모리 정리
            clear_gpu_memory()
        
        _t_surrogate = _sync_time()
        model = get_fitted_model(train_X, train_Y)
        
        C_list_torch = torch.cat(C_list, dim=-1)
        pca_model = PCA(n_components=num_components)
        new_C_list = pca_model.fit_transform(C_list_torch)
        const_model_list = []
        for i in range(num_components):
            const_model_list.append(get_fitted_model(train_X, new_C_list[:, i].unsqueeze(-1)))
            
        _time_surrogate = _sync_time() - _t_surrogate
        clear_gpu_memory()

        _t_candidates = _sync_time()
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            C = torch.cat(C_list, dim = -1)
            const_model_list_gp = ModelListGP(*const_model_list)
            X_next = generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=C,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                constraint_model=const_model_list_gp,
                sobol=sobol,
            )
        
        _time_candidates = _sync_time() - _t_candidates
        del model, const_model_list, const_model_list_gp
        clear_gpu_memory()

        _t_oracle = _sync_time()
        Y_next, C_next, score = test_function.eval_all(X_next)
        _time_oracle = _sync_time() - _t_oracle
        
        # Update TuRBO state
        state = update_state(state=state, Y_next=Y_next, C_next=C_next)

        # Append data
        train_X = torch.cat((train_X, X_next), dim=0)
        train_Y = torch.cat((train_Y, Y_next), dim=0)
        for i in range(num_constraints):
            C_list[i] = torch.cat((C_list[i], C_next[:, i].unsqueeze(-1)), dim=0)
        
        total_X = torch.cat((total_X, X_next), dim=0)
        total_Y = torch.cat((total_Y, Y_next), dim=0)
        total_C = torch.cat((total_C, C_next), dim=0)
        
        scores = torch.cat((scores, score), dim=0)

        print(f"Current Max Score: {score.max():.2e}, Max so far: {scores.max():.2e}")

        # 상태 출력
        if (state.best_constraint_values <= 0).all():
            print(f"{len(total_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
        else:
            violation = state.best_constraint_values.clamp(min=0).sum()
            print(
                f"{len(total_X)}) No feasible point yet! Smallest total violation: "
                f"{violation:.2e}, TR length: {state.length:.2e}"
            )
            
        _time_total = _time_surrogate + _time_candidates + _time_oracle
        print(f"[TIME] Round {iteration}: surrogate={_time_surrogate:.2f}s generative=0.00s candidates={_time_candidates:.2f}s oracle={_time_oracle:.2f}s total={_time_total:.2f}s")

        wandb.log({
            "iteration": iteration,
            "max_score": scores.max().item(),
            "best_value": state.best_value,
            "total_X": len(total_X),
        })

        # 결과 저장
        save_len = min(len(scores), max_evals)
        save_np = scores[:save_len].cpu().numpy()
        file_name = f"pcagp_scbo_{task}_{dim}_{seed}_{n_init}_{indicator}_{batch_size}_{max_evals}_{len(scores)}.npy"
        previous_file_name = f"pcagp_scbo_{task}_{dim}_{seed}_{n_init}_{indicator}_{batch_size}_{max_evals}_{len(scores)-batch_size}.npy"
        save_numpy_array(path=args.save_path, array=save_np, file_name=file_name)
        
        # 이전 파일 삭제
        if os.path.exists(os.path.join(args.save_path, previous_file_name)):
            os.remove(os.path.join(args.save_path, previous_file_name))
        
        # 반복 종료 시 메모리 정리
        clear_gpu_memory()