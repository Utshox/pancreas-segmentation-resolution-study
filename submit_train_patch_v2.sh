#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=patch_unet_v2
#SBATCH --output=%j_patch_unet_v2.log

echo "Starting Patch-Based U-Net Training (v2)..."
hostname
date
nvidia-smi

# Environment Setup
# Try standard module load first
module load python/3.10 2>/dev/null || module load python 2>/dev/null
# module load tensorflow/2.11.0 2>/dev/null # Conflicts with venv

# Manual CUDA paths (since module is missing)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/x86_64-linux-gnu

# If modules fail, try to use the venv from the old account (if readable)
# or assume user has created one.
if [ -d "$HOME/ish/venv_pancreas" ]; then
    source "$HOME/ish/venv_pancreas/bin/activate"
fi

# Define paths
CODE_DIR="$HOME/ish/pancreas-segmentation/src"
DATA_DIR="$HOME/ish/preprocessed_v5_patches"
OUTPUT_DIR="results_patch_unet_v2"

mkdir -p "$OUTPUT_DIR"

echo "Running: run_patch_training_v2.py"
echo "Data: $DATA_DIR"

python "$CODE_DIR/run_patch_training_v2.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Training Complete!"
date
