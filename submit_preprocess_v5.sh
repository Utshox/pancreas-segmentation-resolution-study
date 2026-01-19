#!/bin/bash
#SBATCH -p main
#SBATCH -n 1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=prep_v5
#SBATCH --output=%j_prep_v5.log

echo "Starting Patch Extraction (Stroke Paper Approach)..."
hostname
date

# module load tensorflow/2.11.0 # Failed, using venv instead
module load python
source $HOME/ish/venv_pancreas/bin/activate


# Define paths
CODE_DIR="$HOME/ish/pancreas-segmentation/src/preprocessing"
RAW_DIR="$HOME/ish"
OUTPUT_DIR="$HOME/ish/preprocessed_v5_patches"

echo "Running: preprocess_v5_patches.py"
echo "From: $RAW_DIR"
echo "To: $OUTPUT_DIR"

python "$CODE_DIR/preprocess_v5_patches.py" \
    --raw_dir "$RAW_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Preprocessing Complete!"
date
