
"""
Preprocessing V5: Patch-Based Extraction (The "Stroke Paper" Approach)
----------------------------------------------------------------------
Goal: Extract 256x256 patches from original 512x512 CT volumes.
Strategy:
    - Load original NIfTI volume (512x512).
    - Apply HU Windowing [-125, 275] & Normalize [0, 1].
    - Balance Sampling:
        - 50% patches centered on FOREGROUND (Pancreas).
        - 50% patches from random BACKGROUND.
    - Save as `case_XXX.npy` containing (N, 256, 256) arrays.
    
Target Dataset Size:
    - ~200 patches per volume * 221 patients = ~44,000 patches.
    - 44,000 * 256x256 * 4 bytes ~= 11 GB (Manageable).
"""

import os
import shutil
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_patches(img_vol, label_vol, patch_size=256, n_patches_per_vol=70):
    """
    Extracts balanced patches from a 3D volume.
    Returns:
        patches_x: (N, 256, 256) float32
        patches_y: (N, 256, 256) uint8
    """
    patches_x = []
    patches_y = []
    
    # Identify slices with pancreas
    fg_indices = []
    all_indices = []
    
    num_slices = img_vol.shape[0] # Assuming (D, H, W)
    H, W = img_vol.shape[1], img_vol.shape[2]
    
    # Pre-scan slices
    for i in range(num_slices):
        slice_label = label_vol[i]
        if np.sum(slice_label) > 0:
            fg_indices.append(i)
        all_indices.append(i)
        
    if len(fg_indices) == 0:
        print("Warning: No pancreas found in volume.")
        fg_indices = all_indices # Fallback
        
    num_fg = n_patches_per_vol // 2
    num_bg = n_patches_per_vol - num_fg
    
    # --- Extract Foreground Patches ---
    tries = 0
    while len(patches_x) < num_fg and tries < num_fg * 5:
        tries += 1
        # Random slice with pancreas
        slice_idx = np.random.choice(fg_indices)
        
        # Determine crop center (focus on pancreas)
        ys, xs = np.where(label_vol[slice_idx] > 0)
        if len(ys) > 0:
            center_y = np.random.choice(ys)
            center_x = np.random.choice(xs)
        else:
            center_y = H // 2
            center_x = W // 2
            
        # Add random jitter to center
        jitter = patch_size // 4
        center_y += np.random.randint(-jitter, jitter)
        center_x += np.random.randint(-jitter, jitter)
        
        # Calculate bounding box
        y1 = max(0, center_y - patch_size // 2)
        x1 = max(0, center_x - patch_size // 2)
        
        # Ensure full patch size (handling boundaries)
        if y1 + patch_size > H: y1 = H - patch_size
        if x1 + patch_size > W: x1 = W - patch_size
        y1 = max(0, y1); x1 = max(0, x1); # Handle small images if any
        
        # Crop
        img_patch = img_vol[slice_idx, y1:y1+patch_size, x1:x1+patch_size]
        lab_patch = label_vol[slice_idx, y1:y1+patch_size, x1:x1+patch_size]
        
        # Verify crop size and content (optionally check if empty)
        if img_patch.shape == (patch_size, patch_size):
            patches_x.append(img_patch)
            patches_y.append(lab_patch)

    # --- Extract Background Patches ---
    tries = 0
    while len(patches_x) < n_patches_per_vol and tries < num_bg * 5:
        tries += 1
        slice_idx = np.random.randint(0, num_slices)
        
        # Random top-left corner
        y1 = np.random.randint(0, H - patch_size)
        x1 = np.random.randint(0, W - patch_size)
        
        img_patch = img_vol[slice_idx, y1:y1+patch_size, x1:x1+patch_size]
        lab_patch = label_vol[slice_idx, y1:y1+patch_size, x1:x1+patch_size]
        
        if img_patch.shape == (patch_size, patch_size):
            patches_x.append(img_patch)
            patches_y.append(lab_patch)
            
    return np.array(patches_x), np.array(patches_y)

def process_case(case_folder, output_dir, img_dir, label_dir):
    try:
        case_id = case_folder.name.replace('.nii.gz', '') # Handle case where input is file or folder logic
        # Filename logic: pancreas_XXX.nii.gz
        
        img_path = img_dir / f"{case_id}.nii.gz"
        lab_path = label_dir / f"{case_id}.nii.gz"
        
        if not img_path.exists() or not lab_path.exists():
            return
            
        # Load NIfTI
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lab = nib.load(str(lab_path)).get_fdata().astype(np.uint8)
        
        # Standardize Orientation (Assume (H, W, D) coming in -> (D, H, W) is better for slicing)
        # NIfTI default is usually (H, W, D)
        img = np.transpose(img, (2, 0, 1)) # -> (D, H, W)
        lab = np.transpose(lab, (2, 0, 1))
        
        # Verify 512x512
        if img.shape[1] != 512 or img.shape[2] != 512:
            print(f"Skipping {case_id}: non-standard size {img.shape}")
            return
            
        # --- Preprocessing (Windowing) ---
        HU_MIN, HU_MAX = -125, 275
        img = np.clip(img, HU_MIN, HU_MAX)
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)
        img = np.clip(img, 0.0, 1.0)
        
        # --- Patch Extraction ---
        patches_x, patches_y = extract_patches(img, lab)
        
        # Save as .npy
        np.save(output_dir / f"{case_id}_x.npy", patches_x.astype(np.float32))
        np.save(output_dir / f"{case_id}_y.npy", patches_y.astype(np.uint8))
        
    except Exception as e:
        print(f"Error processing {case_folder.name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help="Path containing imagesTr/ and labelsTr/")
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    
    img_dir = raw_path / 'imagesTr'
    label_dir = raw_path / 'labelsTr'
    
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    
    # Get case list
    cases = sorted(list(img_dir.glob('*.nii.gz')))
    print(f"Found {len(cases)} cases. Extracting 200 patches/case...")
    
    for case_file in tqdm(cases):
        process_case(case_file, out_path, img_dir, label_dir)
        
if __name__ == "__main__":
    main()
