export PYTHONPATH=/home/uky/repos_python/Release/CiBO/baselines:$PYTHONPATH


# ----- Indicator False -----
# Ackley
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 3000 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait

# Rastrigin
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait

#Rosenbrock
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait



# ----- Indicator True -----
#Ackley
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 3000 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator true --M 10\
        &
done
wait

# Rastrigin
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator true --M 10\
        &
done
wait

#Rosenbrock
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 10 --buffer_size 2000 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator true --M 10\
        &
done
wait

# ----- Realworld -----
# RoverPlanning
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task RoverPlanning --dim 60 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 3 --buffer_size 2000 --flow_steps 250 --alpha 1e-4 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait


# MOPTA
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task Mopta --dim 124 --batch_size 20\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 3 --buffer_size 500 --flow_steps 250 --alpha 1e-5 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 10\
        &
done
wait

# DNA
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/CiBO.py --wandb true --task DNA --dim 180 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --proxy_hidden_dim 1024 --prior_hidden_dim 512 --num_proxy_epochs 100 --num_prior_epochs 500 --num_posterior_epochs 50\
                --lamb 5 --buffer_size 1000 --flow_steps 250 --alpha 1e-3 --beta 1 --t_scale 1.0\
                --hidden_dim 256 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 --gfn_buffer_size 600000 --gfn_batch_size 500\
                --T 50  --pis_architectures --zero_init --clipping --mode_fwd tb  --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
                --both_ways --prioritized rank --rank_weight 1e-2 --indicator false --M 15\
        &
done
wait