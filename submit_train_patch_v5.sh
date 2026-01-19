#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=patch_unet_v5
#SBATCH --output=%j_patch_unet_v5.log

echo "Starting Patch-Based U-Net Training (v5 - Final GPU Fix)..."
hostname
date
nvidia-smi

# Environment Setup
if [ -d "$HOME/ish/venv_pancreas" ]; then
    source "$HOME/ish/venv_pancreas/bin/activate"
fi

# DYNAMIC GPU CONFIGURATION
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"

# 1. Add all nvidia libs to LD_LIBRARY_PATH
# This finds every 'lib' folder under nvidia/ (nvcc, runtime, cudnn, etc)
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS

# 2. Tell XLA where to find libdevice.10.bc
# typically in nvidia/cuda_nvcc/lib/ or share/
NVCC_LIB_DIR=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1 | xargs dirname)
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(dirname $NVCC_LIB_DIR)

echo "--- Debug Info ---"
echo "LD_LIBRARY_PATH includes: $NVIDIA_LIBS"
echo "XLA_FLAGS: $XLA_FLAGS"
echo "libdevice found at: $NVCC_LIB_DIR"

# Define paths
CODE_DIR="$HOME/ish/pancreas-segmentation/src"
DATA_DIR="$HOME/ish/preprocessed_v5_patches"
OUTPUT_DIR="results_patch_unet_v5"

mkdir -p "$OUTPUT_DIR"

echo "Running: run_patch_training_v2.py"

python "$CODE_DIR/run_patch_training_v2.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Training Complete!"
date
