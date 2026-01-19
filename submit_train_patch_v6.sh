#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=patch_unet_v6
#SBATCH --output=%j_patch_unet_v6.log

echo "Starting Patch-Based U-Net Training (v6 - Fixed XLA Path)..."
hostname
date
nvidia-smi

# Environment Setup
if [ -d "$HOME/ish/venv_pancreas" ]; then
    source "$HOME/ish/venv_pancreas/bin/activate"
fi

# DYNAMIC GPU CONFIGURATION
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"

# 1. LD_LIBRARY_PATH (Same as v4/v5 - correct)
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS

# 2. XLA FLAGS (The Fix)
# File is at: .../nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
# XLA expects ROOT such that ROOT/nvvm/libdevice/libdevice.10.bc exists.
# So we need ROOT = .../nvidia/cuda_nvcc

LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
# Go up 3 levels: file -> libdevice/ -> nvvm/ -> cuda_nvcc/
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

echo "--- Debug Info ---"
echo "Found libdevice at: $LIBDEVICE_PATH"
echo "Calculated CUDA_ROOT: $CUDA_ROOT"
echo "XLA_FLAGS: $XLA_FLAGS"

# Define paths
CODE_DIR="$HOME/ish/pancreas-segmentation/src"
DATA_DIR="$HOME/ish/preprocessed_v5_patches"
OUTPUT_DIR="results_patch_unet_v6"

mkdir -p "$OUTPUT_DIR"

echo "Running: run_patch_training_v2.py"

python "$CODE_DIR/run_patch_training_v2.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Training Complete!"
date
