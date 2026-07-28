export PYTHONPATH=/path/to/CiBO:/path/to/CiBO/baselines:$PYTHONPATH

# Hyperparameters follow Appendix D.3 of the paper.
#   The inverse temperature beta of Eq. (12) is implemented as --alpha = 1 / beta with --beta 1,
#   so beta = 1e5 corresponds to --alpha 1e-5 and is fixed for every task.
#   --lamb = lambda (Table 4), --buffer_size = L, --rank_k = kappa, --M = N.
# Seeds follow the paper: 12 seeds for synthetic tasks, 4 seeds for real-world tasks.
# Synthetic runs are dispatched over 4 GPUs, 4 seeds at a time.

# ----- Indicator False -----
# Ackley
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/cibo.py --wandb false --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --rank_k 1e-2 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rastrigin
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/cibo.py --wandb false --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --rank_k 1e-2 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

#Rosenbrock
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/cibo.py --wandb false --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --rank_k 1e-2 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait



# ----- Indicator True -----
#Ackley
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/cibo.py --wandb false --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --rank_k 1e-2 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator true --M 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rastrigin
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/cibo.py --wandb false --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --rank_k 1e-2 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator true --M 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

#Rosenbrock
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/cibo.py --wandb false --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --rank_k 1e-2 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator true --M 10\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# ----- Realworld -----
# RoverPlanning
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/cibo.py --wandb false --task RoverPlanning --dim 60 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 5 --buffer_size 1000 --rank_k 1e-1 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait


# HalfCheetah
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/cibo.py --wandb false --task HalfCheetah --dim 102 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 5 --buffer_size 1000 --rank_k 1e-1 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait


# MOPTA
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/cibo.py --wandb false --task Mopta --dim 124 --batch_size 20\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 1000 --rank_k 1e-1 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait

# DNA
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/cibo.py --wandb false --task DNA --dim 180 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 5 --buffer_size 1000 --rank_k 1e-1 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait
