export PYTHONPATH=/path/to/CiBO:/path/to/CiBO/baselines:$PYTHONPATH

# DiffOPT (Kong et al.), re-implemented as described in Appendix C of the paper.
# Seeds follow the paper: 12 seeds for synthetic tasks, 4 seeds for real-world tasks.
# Synthetic runs are dispatched over 4 GPUs, 4 seeds at a time.
# There is no public implementation; the data distribution is modelled with a diffusion model
# of similar architecture to CiBO, and candidates are drawn with Langevin dynamics on the
# differentiable surrogate energy. The reported runs use lambda = 10 and a buffer of 1000 for the
# synthetic tasks, and lambda = 3 with a buffer of 500 for the real-world ones.

# ----- Indicator False -----
# Ackley
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/diffopt.py --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rastrigin
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/diffopt.py --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rosenbrock
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/diffopt.py --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# ----- Indicator True -----
# Ackley
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/diffopt.py --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator true --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rastrigin
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/diffopt.py --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator true --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rosenbrock
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/diffopt.py --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator true --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# ----- Realworld -----
# RoverPlanning
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/diffopt.py --task RoverPlanning --dim 60 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 3 --buffer_size 500\
        &
done
wait

# HalfCheetah
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/diffopt.py --task HalfCheetah --dim 102 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 3 --buffer_size 500\
        &
done
wait

# Mopta
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/diffopt.py --task Mopta --dim 124 --batch_size 20\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 3 --buffer_size 500\
        &
done
wait

# DNA
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/diffopt.py --task DNA --dim 180 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --num_proxy_epochs 50 --num_prior_epochs 50 --diffusion_steps 30 --constraint_formulation Lagrangian --lamb 3 --buffer_size 500\
        &
done
wait
