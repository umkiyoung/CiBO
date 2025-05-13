# Posterior Inference in Latent Space for Scalable Constrained Black-box Optimization (CiBO)

## Installation Guide

To ensure that the CiBO repository is included in your Python path, add the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`):
```bash
nano ~./bashrc #open shell configuration
```
```bash
#Example:
export PYTHONPATH=/home/name/CiBO:$PYTHONPATH
```

After adding the line, reload the shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

Or add lines ```export PYTHONPATH=/home/name/CiBO:$PYTHONPATH```  the top of the [baselines/scripts/cibo](baselines/scripts/cibo.sh) files.

### Environment settings
```bash
# Create conda environment
conda create -n cibo python=3.9 -y
conda activate cibo

# Mujoco Installation (Mujoco should be already placed in ~/.mujoco)
pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
pip3 install box2d-py Box2D
# Mujoco Compile
python -c "import mujoco_py"

# Torch Installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Additional Dependencies
pip install botorch==0.6.4 gpytorch==1.6.0
pip install gym==0.13.1 attrdict==2.0.1 wandb==0.15.3 matplotlib==3.7.5
pip install pandas==1.5.3 scikit-learn==1.2.2 tqdm==4.64.1 
pip install torchdiffeq

# Lasso Env
pip install celer
pip install "sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip"
pip install libsvmdata
pip install pygame

pip install einops
pip install POT
```

# Run examples:

```bash
sh baselines/scripts/cibo.sh
```

All the settings are written in the [baselines/scripts](baselines/scripts) folder.

# References:
Our implementation on diffusion sampler is based on "Improved Improved off-policy training of diffusion samplers" (https://github.com/GFNOrg/gfn-diffusion)

