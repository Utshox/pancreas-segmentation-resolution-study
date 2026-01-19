#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=patch_unet_v3
#SBATCH --output=%j_patch_unet_v3.log

echo "Starting Patch-Based U-Net Training (v3 - Pip GPU Libs)..."
hostname
date
nvidia-smi

# Environment Setup
# No modules needed - we use venv with bundled libs!
module load python/3.10 2>/dev/null || module load python 2>/dev/null

# If modules fail, try to use the venv from the old account
if [ -d "$HOME/ish/venv_pancreas" ]; then
    source "$HOME/ish/venv_pancreas/bin/activate"
fi

# Manual CUDA paths -> Point to the Pip Installed Libs inside venv!
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cuda_runtime/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$VENV_SITE/nvidia/cuda_runtime

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Define paths
CODE_DIR="$HOME/ish/pancreas-segmentation/src"
DATA_DIR="$HOME/ish/preprocessed_v5_patches"
OUTPUT_DIR="results_patch_unet_v3"

mkdir -p "$OUTPUT_DIR"

echo "Running: run_patch_training_v2.py"
echo "Data: $DATA_DIR"

python "$CODE_DIR/run_patch_training_v2.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Training Complete!"
date
