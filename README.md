# Pancreas Segmentation Resolution Study

This repository contains the code and resources for the study "High-Resolution Patch-Based Semi-Supervised Learning for Pancreas Segmentation".

## 🏆 Key Findings
*   **Resolution is Key:** Maintaining 512x512 native resolution (via patching) boosts Dice from 0.73 to **0.85**.
*   **Mean Teacher Success:** Using Mean Teacher SSL with 50% data achieves **0.83 Dice**, significantly outperforming FixMatch (0.69).

## 📂 Repository Structure
*   `src/`: Core implementation code.
    *   `run_patch_training.py`: Supervised training loop.
    *   `run_patch_meanteacher.py`: Semi-Supervised Mean Teacher training.
    *   `inference/`: Sliding window inference scripts.
    *   `visualization/`: Scripts to generate qualitative plots.
*   `paper/`: Conference paper (LaTeX) and visualizations.
    *   `walkthrough.md`: **Detailed Project Report & Architecture Explanation**.

## 🚀 How to Run
### 1. Preprocessing
```bash
python src/preprocessing/preprocess_v5_patches.py
```
### 2. Training (Supervised)
```bash
python src/run_patch_training.py --batch_size 16
```
### 3. Training (Mean Teacher SSL)
```bash
python src/run_patch_meanteacher.py --labeled_ratio 0.5
```

## 📊 Results Summary
| Method | Dice Score |
|--------|------------|
| Standard U-Net (Resized) | 0.73 |
| **High-Res Patch U-Net** | **0.85** |
| Mean Teacher (50% Data) | 0.83 |
