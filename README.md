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
pip install botorch==0.6.4 gpytorch==1.6.0
pip install gym==0.13.1 attrdict==2.0.1 wandb==0.15.3 matplotlib==3.7.5
pip install pandas==1.5.3 scikit-learn==1.2.2 tqdm==4.64.1 
pip install torchdiffeq

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

Runs are launched round-robin over the available GPUs, `NUM_GPUS` at a time (default 4):

```bash
NUM_GPUS=8 sh baselines/scripts/cibo.sh
```

The HalfCheetah task additionally requires `gym` and MuJoCo (see [Environment Setup](#-environment-setup)).

---

## 📚 References

Our implementation of the diffusion sampler is based on:
- [Improved Improved off-policy training of diffusion samplers](https://github.com/GFNOrg/gfn-diffusion)

---

Feel free to open issues or pull requests for questions, suggestions, or contributions!

