#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=patch_fix
#SBATCH --output=%j_patch_fix.log

echo "Starting Patch-Based FixMatch (SSL)..."
hostname
date
nvidia-smi

# Environment Setup
if [ -d "$HOME/ish/venv_pancreas" ]; then
    source "$HOME/ish/venv_pancreas/bin/activate"
fi

# DYNAMIC GPU CONFIGURATION (Same as v6)
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"

# 1. LD_LIBRARY_PATH
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS

# 2. XLA FLAGS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

echo "Environment Configured."

# Define paths
CODE_DIR="$HOME/ish/pancreas-segmentation/src"
DATA_DIR="$HOME/ish/preprocessed_v5_patches"
OUTPUT_DIR="results_patch_fixmatch"

mkdir -p "$OUTPUT_DIR"

echo "Running: run_patch_fixmatch.py"
echo "Data: $DATA_DIR"

# Run with 10% Labeled Data
python "$CODE_DIR/run_patch_fixmatch.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --labeled_ratio 0.1

echo "Training Complete!"
date
