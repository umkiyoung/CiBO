#!/usr/bin/env sh
# Reproduction script for CiBO.
#
# Seeds follow the paper: 12 random seeds for synthetic tasks, 4 for real-world tasks.
# Hyperparameters follow Appendix D.3:
#   * The inverse temperature beta of Eq. (12) is implemented as --alpha = 1 / beta
#     with --beta fixed to 1, so beta = 1e5 corresponds to --alpha 1e-5 (fixed for all tasks).
#   * --lamb = lambda (Table 4), --buffer_size = L, --rank_k = kappa, --M = N, --gamma = 1.
#
# Usage:  sh baselines/scripts/cibo.sh          (override GPU count with NUM_GPUS=8 sh ...)

CIBO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH=$CIBO_ROOT:$CIBO_ROOT/baselines:$PYTHONPATH
cd "$CIBO_ROOT" || exit 1

NUM_GPUS=${NUM_GPUS:-4}
SYNTHETIC_SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"   # 12 seeds (Figure 3)
REALWORLD_SEEDS="0 1 2 3"                     # 4 seeds  (Figure 4)

# Settings shared by every task (Tables 1-3, Appendix D.3).
COMMON="--wandb true --n_init 200 --gamma 1 --M 10 \
        --proxy_hidden_dim 1024 --prior_hidden_dim 512 \
        --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50 \
        --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0 \
        --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 \
        --gfn_buffer_size 600000 --gfn_batch_size 500 --T 50 \
        --pis_architectures --zero_init --clipping --mode_fwd tb \
        --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1 \
        --both_ways --prioritized rank --rank_weight 1e-2"

# run_task <seed list> <extra args...>
# Launches one run per seed, round-robin over NUM_GPUS, NUM_GPUS runs at a time.
run_task() {
    seeds=$1
    shift
    i=0
    for seed in $seeds; do
        CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) \
            python baselines/algorithms/CiBO.py --seed "$seed" "$@" &
        i=$((i + 1))
        if [ $((i % NUM_GPUS)) -eq 0 ]; then wait; fi
    done
    wait
}

# ---------------------------------------------------------------------------
# Synthetic tasks: |D0| = 200, B = 100, 10,000 evaluations, lambda = 10, L = 2000, kappa = 0.01
# ---------------------------------------------------------------------------
for task in Ackley Rastrigin Rosenbrock; do
    run_task "$SYNTHETIC_SEEDS" --task "$task" --dim 200 --batch_size 100 --max_evals 10000 \
        --lamb 10 --buffer_size 2000 --rank_k 1e-2 --indicator false $COMMON
done

# Indicator constraint setting (binary feasibility feedback only)
for task in Ackley Rastrigin Rosenbrock; do
    run_task "$SYNTHETIC_SEEDS" --task "$task" --dim 200 --batch_size 100 --max_evals 10000 \
        --lamb 10 --buffer_size 2000 --rank_k 1e-2 --indicator true $COMMON
done

# ---------------------------------------------------------------------------
# Real-world tasks: |D0| = 200, 2,000 evaluations, L = 1000, kappa = 0.1
# B = 50 for all tasks except Mopta (B = 20).
# ---------------------------------------------------------------------------
# RoverPlanning (D = 60, 15 constraints)
run_task "$REALWORLD_SEEDS" --task RoverPlanning --dim 60 --batch_size 50 --max_evals 2000 \
    --lamb 5 --buffer_size 1000 --rank_k 1e-1 --indicator false $COMMON

# HalfCheetah (D = 102, 5 constraints) -- requires gym + mujoco (see README)
run_task "$REALWORLD_SEEDS" --task HalfCheetah --dim 102 --batch_size 50 --max_evals 2000 \
    --lamb 5 --buffer_size 1000 --rank_k 1e-1 --indicator false $COMMON

# Mopta (D = 124, 68 constraints)
run_task "$REALWORLD_SEEDS" --task Mopta --dim 124 --batch_size 20 --max_evals 2000 \
    --lamb 10 --buffer_size 1000 --rank_k 1e-1 --indicator false $COMMON

# Lasso DNA (D = 180, 5 constraints)
run_task "$REALWORLD_SEEDS" --task DNA --dim 180 --batch_size 50 --max_evals 2000 \
    --lamb 5 --buffer_size 1000 --rank_k 1e-1 --indicator false $COMMON
