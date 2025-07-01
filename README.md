# 🚀 Posterior Inference in Latent Space for Scalable Constrained Black-box Optimization (CiBO)

A scalable framework for constrained black-box optimization using posterior inference in latent space.

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

---

## 📚 References

Our implementation of the diffusion sampler is based on:
- [Improved Improved off-policy training of diffusion samplers](https://github.com/GFNOrg/gfn-diffusion)

---

Feel free to open issues or pull requests for questions, suggestions, or contributions!

