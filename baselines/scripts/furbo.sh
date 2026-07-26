export PYTHONPATH=/home/uky/repos_python/Release/CiBO:/home/uky/repos_python/Release/CiBO/baselines:$PYTHONPATH

# FuRBO (Ascia et al., 2025).
# Seeds follow the paper: 12 seeds for synthetic tasks, 4 seeds for real-world tasks.
# Synthetic runs are dispatched over 4 GPUs, 4 seeds at a time.


# ----- Indicator False -----
# Ackley
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/furbo.py --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rastrigin
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/furbo.py --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rosenbrock
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/furbo.py --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# ----- Indicator True -----
# Ackley
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/furbo.py --task Ackley --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator true --inspector_percentage 0.1\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rastrigin
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/furbo.py --task Rastrigin --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator true --inspector_percentage 0.1\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# Rosenbrock
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
        CUDA_VISIBLE_DEVICES=$((seed % 4)) python baselines/algorithms/furbo.py --task Rosenbrock --dim 200 --batch_size 100\
                --n_init 200 --max_evals 10000 --seed $seed --indicator true --inspector_percentage 0.1\
        &
        if [ $(( (seed + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait

# ----- Realworld -----
# RoverPlanning
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/furbo.py --task RoverPlanning --dim 60 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
done
wait

# HalfCheetah
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/furbo.py --task HalfCheetah --dim 102 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
done
wait

# Mopta
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/furbo.py --task Mopta --dim 124 --batch_size 20\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
done
wait

# DNA
for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed python baselines/algorithms/furbo.py --task DNA --dim 180 --batch_size 50\
                --n_init 200 --max_evals 2000 --seed $seed --indicator false --inspector_percentage 0.1\
        &
done
wait
