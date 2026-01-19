
"""
Sliding Window Inference for Patch-Based U-Net
----------------------------------------------
Reconstructs 512x512 segmentation maps from 256x256 patch predictions.
"""

import numpy as np
import tensorflow as tf
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm

def predict_sliding_window(model, image_vol, patch_size=256, stride=128, batch_size=8):
    """
    Predicts a 3D volume using sliding window 2D patches.
    image_vol: (D, H, W)
    Returns: probability_map (D, H, W)
    """
    D, H, W = image_vol.shape
    prob_map = np.zeros_like(image_vol, dtype=np.float32)
    count_map = np.zeros_like(image_vol, dtype=np.float32)
    
    # Generate patch coordinates
    y_starts = np.arange(0, H - patch_size + 1, stride)
    x_starts = np.arange(0, W - patch_size + 1, stride)
    
    # Handle edges if not divisible by stride
    if y_starts[-1] + patch_size < H:
        y_starts = np.append(y_starts, H - patch_size)
    if x_starts[-1] + patch_size < W:
        x_starts = np.append(x_starts, W - patch_size)
        
    y_starts = np.unique(y_starts)
    x_starts = np.unique(x_starts)
    
    # Prepare batch of patches
    patches = []
    coords = []
    
    for d in range(D):
        slice_img = image_vol[d]
        
        for y in y_starts:
            for x in x_starts:
                patch = slice_img[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coords.append((d, y, x))
                
    # Predict in batches
    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=-1) # (N, 256, 256, 1)
    
    preds = model.predict(patches, batch_size=batch_size, verbose=0)
    
    # Reconstruct
    for i, (d, y, x) in enumerate(coords):
        pred_patch = preds[i, :, :, 0]
        prob_map[d, y:y+patch_size, x:x+patch_size] += pred_patch
        count_map[d, y:y+patch_size, x:x+patch_size] += 1.0
        
    # Average
    return prob_map / np.maximum(count_map, 1.0)

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True, help="Path to imagesTr")
    parser.add_argument('--label_dir', type=str, required=True, help="Path to labelsTr")
    parser.add_argument('--num_cases', type=int, default=5, help="Number of cases to validate")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    test_cases = sorted(list(Path(args.image_dir).glob('*.nii.gz')))[:args.num_cases]
    scores = []
    
    print(f"Running inference on {len(test_cases)} cases...")
    for case_path in tqdm(test_cases):
        # Load
        img = nib.load(str(case_path)).get_fdata().astype(np.float32)
        # Transpose to (D, H, W) if needed
        img = np.transpose(img, (2, 0, 1))
        
        # Preprocess (Windowing)
        img = np.clip(img, -125, 275)
        img = (img + 125) / 400.0
        
        # Predict
        probs = predict_sliding_window(model, img)
        pred_mask = (probs > 0.5).astype(np.uint8)
        
        # Load GT
        case_id = case_path.name
        lbl_path = Path(args.label_dir) / case_id
        gt = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
        gt = np.transpose(gt, (2, 0, 1))
        
        score = dice_score(gt, pred_mask)
        scores.append(score)
        print(f"{case_id}: Dice = {score:.4f}")
        
    print(f"Average Dice: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
