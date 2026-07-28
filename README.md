# 🚀 Posterior Inference in Latent Space for Scalable Constrained Black-box Optimization (CiBO)

This repository implements the CiBO method presented in our paper **"Posterior Inference in Latent Space
for Scalable Constrained Black-box Optimization"**. CiBO leverages diffusion models to perform efficient posterior inference and amortize the sampling from the posterior distribution in the latent space
of flow-based models, achieving superior performance on various **synthetic and real-world constrained black-box optimization** tasks. 

For a visual explanation of the CiBO framework, please see the image below:
![figure1_update_ver4](https://github.com/user-attachments/assets/3eed0a81-1e14-45f3-9667-161f1c726cf2)

---
## 📑 Table of Contents
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Running Examples](#running-examples)
- [References](#references)

---

## 🛠️ Installation

To include the CiBO repository in your Python path, add the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`):

```bash
# Open your shell configuration file
nano ~/.bashrc  # or nano ~/.zshrc

# Add the following line (replace /home/name/CiBO with your path)
export PYTHONPATH=/home/name/CiBO:$PYTHONPATH
```

After editing, reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

Alternatively, add the same export line to the top of [`baselines/scripts/cibo.sh`](baselines/scripts/cibo.sh).

---

## 🧩 Environment Setup

We recommend using [conda](https://docs.conda.io/en/latest/) for environment management.

```bash
# 1. Create and activate conda environment
conda create -n cibo python=3.9 -y
conda activate cibo

# 2. Mujoco Installation (Mujoco should be in ~/.mujoco)
pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
pip install box2d-py Box2D
python -c "import mujoco_py"  # Mujoco compile test

# 3. Torch Installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 4. Additional Dependencies
pip install botorch gpytorch          # >= 0.8 (>= 0.9 for the LogcEI baseline)
pip install gym==0.13.1 attrdict==2.0.1 wandb==0.15.3 matplotlib==3.7.5
pip install pandas==1.5.3 scikit-learn==1.2.2 tqdm==4.64.1 
pip install torchdiffeq

# 4b. Baseline-only dependencies
pip install cma           # CMA-ES
pip install torch_pca     # PCAGP-SCBO
pip install scipy         # COBYLA

# 5. Lasso Environment
pip install celer
pip install "sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip"
pip install libsvmdata
pip install pygame

# 6. Other dependencies
pip install einops POT
```

---

## ▶️ Running Examples

To run the main CiBO example:

```bash
sh baselines/scripts/cibo.sh
```

All configuration settings are available in the [`baselines/scripts`](baselines/scripts) folder.

### Reproducing the results in the paper

`cibo.sh` runs every task reported in the paper with the hyperparameters of Appendix D.3:

| | Tasks | Seeds | \|D0\| | B | Budget | λ | L | κ |
|---|---|---|---|---|---|---|---|---|
| Synthetic | Ackley / Rastrigin / Rosenbrock (200D), each with and without the indicator constraint | 12 (`0`–`11`) | 200 | 100 | 10,000 | 10 | 2000 | 0.01 |
| Real-world | RoverPlanning (60D), HalfCheetah (102D), Mopta (124D), Lasso DNA (180D) | 4 (`0`–`3`) | 200 | 50 (Mopta: 20) | 2,000 | 5 / 5 / 10 / 5 | 1000 | 0.1 |

The inverse temperature β of Eq. (12) is fixed to `1e5` for every task. It is implemented as
`--alpha = 1 / β` with `--beta 1`, so the scripts pass `--alpha 1e-5`; `--gamma 1` and `--M 10`
(the N of Appendix D.3) are likewise shared by all tasks.

Each task is launched one seed per GPU, four seeds at a time; adjust the `CUDA_VISIBLE_DEVICES`
assignment in the script if a different number of GPUs is available.

The HalfCheetah task additionally requires `gym` and MuJoCo (see [Environment Setup](#-environment-setup)).

### Baselines

Every baseline compared against in the paper is included, with one script per method covering the
same tasks, seeds and budgets as `cibo.sh`:

| Baseline | Implementation | Script | Notes |
|---|---|---|---|
| SCBO | `baselines/algorithms/scbo.py` | `baselines/scripts/scbo.sh` | |
| PCAGP-SCBO | `baselines/algorithms/pcagp_scbo.py` | `baselines/scripts/pcagp_scbo.sh` | needs `torch_pca`; `l = 2` on synthetic tasks, searched over `[2, ⌊D/2⌋]` on real-world tasks |
| FuRBO | `baselines/algorithms/furbo.py` | `baselines/scripts/furbo.sh` | |
| DiffOPT | `baselines/algorithms/diffopt.py` | `baselines/scripts/diffopt.sh` | re-implemented (no public code); diffusion model + Langevin dynamics on the surrogate energy |
| cEI | `baselines/algorithms/cei.py` | `baselines/scripts/cei.sh` | `--acqf_name cEI` |
| LogcEI | `baselines/algorithms/cei.py` | `baselines/scripts/logcei.sh` | `--acqf_name logcEI`; needs botorch >= 0.9 |
| CMA-ES | `baselines/algorithms/cmaes.py` | `baselines/scripts/cmaes.sh` | needs `cma`; same Lagrangian and `--lamb` as CiBO |
| COBYLA | `baselines/algorithms/cobyla.py` | `baselines/scripts/cobyla.sh` | `scipy.optimize.minimize(method="COBYLA")` |
| Sobol | `baselines/algorithms/sobol.py` | `baselines/scripts/sobol.sh` | non-adaptive space-filling baseline |

```bash
sh baselines/scripts/scbo.sh      # and likewise for the other baselines
```

Every method writes one `.npy` per run to `baselines/results/<method>/`, holding the running best
feasible score over the evaluation budget (`-inf` until the first feasible point). The mean/std
curves in the paper are the mean and one standard deviation of these arrays across seeds.

Weights & Biases logging is off by default for every method; pass `--wandb true` to enable it.

### Reproducing the paper figures

The runs behind the paper are checked into this repository, so the figures can be regenerated
without re-running any optimization. `baselines/results/` holds 88 `.npy` files per method — the
6 synthetic settings x 12 seeds and the 4 real-world tasks x 4 seeds of the table above:

```bash
python baselines/reproduce_figure3.py   # → baselines/figures/synthetic.pdf   (Figure 3)
python baselines/reproduce_figure4.py   # → baselines/figures/realworld.pdf   (Figure 4)
```

Both scripts only need `numpy` and `matplotlib`, and print `File not found: ...` for any run that
is missing instead of failing, so a partial `results/` directory still plots. The figures use
Times New Roman; without that font installed, matplotlib silently falls back to its default serif
and the output is very slightly different from the version committed under `baselines/figures/`.

#### Memory note for SCBO on Mopta

Mopta has 68 constraints, and SCBO draws its Thompson samples from the joint posterior of all
constraint GPs over `N_CANDIDATES = 5000` points in `float64` (`baselines/algorithms/scbo.py`).
That densifies a 5000 x 5000 block per constraint, so the peak GPU memory is roughly 45 GB —
more than a 48 GB card can hold. Either use a larger GPU, or run this one configuration on CPU:

```bash
CUDA_VISIBLE_DEVICES="" python baselines/algorithms/scbo.py --task Mopta --dim 124 ...
```

Every other (method, task) pair fits comfortably on a single 48 GB GPU.

---

## 📚 References

Our implementation of the diffusion sampler is based on:
- [Improved Improved off-policy training of diffusion samplers](https://github.com/GFNOrg/gfn-diffusion)

---

Feel free to open issues or pull requests for questions, suggestions, or contributions!

