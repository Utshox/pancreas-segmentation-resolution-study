#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=ssl_inf_v2
#SBATCH --output=%j_ssl_inf_v2.log

echo "Starting SSL (FixMatch) Sliding Window Inference (v2 - Fixed Weights)..."
hostname
date
nvidia-smi

# Environment Setup
if [ -d "$HOME/ish/venv_pancreas" ]; then
    source "$HOME/ish/venv_pancreas/bin/activate"
fi

# DYNAMIC GPU CONFIGURATION (Same as v6)
VENV_SITE="$HOME/ish/venv_pancreas/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find $VENV_SITE/nvidia -name "lib" -type d | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
LIBDEVICE_PATH=$(find $VENV_SITE/nvidia -name "libdevice.10.bc" | head -1)
CUDA_ROOT=$(dirname $(dirname $(dirname $LIBDEVICE_PATH)))
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

echo "Environment Configured."

# Define paths
CODE_DIR="$HOME/ish"
DATA_DIR="$HOME/ish/data_val"
MODEL_PATH="$HOME/results_patch_fixmatch/model_fixmatch_best.h5"
OUTPUT_DIR="results_inference_fixmatch_v2"

mkdir -p "$OUTPUT_DIR"

echo "Running Inference on: $DATA_DIR"
echo "Model: $MODEL_PATH"

# Call the NEW script
python "$CODE_DIR/sliding_window_inference_fixmatch.py" \
    --image_dir "$DATA_DIR/imagesTr" \
    --label_dir "$DATA_DIR/labelsTr" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "Inference Complete!"
cat "$OUTPUT_DIR/final_dice.txt"
date
