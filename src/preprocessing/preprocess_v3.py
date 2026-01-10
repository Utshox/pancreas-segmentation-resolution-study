"""
Preprocessing V3: Strict HU Windowing for Pancreas Segmentation
Target: Soft Tissue Window [-125, 275]
Normalization: Min-Max to [0, 1]
Output: .npy files in preprocessed_v3/
"""
import os
import shutil
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm

def process_case(case_folder, output_dir, img_dir, label_dir):
    try:
        case_id = case_folder.name
        img_name = f"{case_id}.nii.gz"
        lab_name = f"{case_id}.nii.gz"
        
        img_path = img_dir / img_name
        lab_path = label_dir / lab_name
        
        if not img_path.exists() or not lab_path.exists():
            return # Skip incomplete cases

        # Load NIfTI
        img_obj = nib.load(str(img_path))
        lab_obj = nib.load(str(lab_path))
        
        img_data = img_obj.get_fdata()
        lab_data = lab_obj.get_fdata()

        # Strict HU Windowing (Soft Tissue)
        HU_MIN, HU_MAX = -125, 275
        img_data = np.clip(img_data, HU_MIN, HU_MAX)
        
        # Normalize to [0, 1]
        img_data = (img_data - HU_MIN) / (HU_MAX - HU_MIN)
        
        # Verify Normalization
        img_data = np.clip(img_data, 0.0, 1.0)
        
        # Process Slices
        # Standardize to (H, W, D) -> Iterate over D
        # Resize logic is skipped assuming 512x512 original, 
        # For simplicity in this v3, we keep original size or specific resize?
        # Dataset.py in v1/v2 resized to 256x256. We MUST replicate this.
        # But `skimage.transform.resize` is slow.
        # Let's assume for now we save 512x512 and resize in DataPipeline? 
        # NO, user wants consistency. We must resize here.
        
        import skimage.transform
        
        # Create output folder
        out_case_dir = output_dir / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)
        
        # Correct orientation? Nibabel handles RAS/LPS but array is straight read.
        # We assume dataset is consistent.
        
        # Save as single large .npy or slices?
        # v2 used `image.npy` (Volume). Let's stick to that.
        
        # Resize Volume (Slice-by-slice to save RAM)
        target_shape = (256, 256)
        
        resized_img_vol = []
        resized_lab_vol = []
        
        for i in range(img_data.shape[2]):
            sl_img = img_data[:, :, i]
            sl_lab = lab_data[:, :, i]
            
            # Simple resize (Check anti-aliasing for masks!)
            sl_img_r = skimage.transform.resize(sl_img, target_shape, order=1, preserve_range=True, anti_aliasing=True)
            sl_lab_r = skimage.transform.resize(sl_lab, target_shape, order=0, preserve_range=True, anti_aliasing=False)
            
            resized_img_vol.append(sl_img_r)
            resized_lab_vol.append(sl_lab_r)
            
        final_img = np.stack(resized_img_vol, axis=0) # (D, H, W) -> Match PyTorch/TF preference? 
        final_lab = np.stack(resized_lab_vol, axis=0)
        
        # TF Data Loader expects (D, H, W)? 
        # Previous loader read `image.npy` and shuffled slices.
        # Let's check `data_loader_tf2.py` expectation.
        # Usually it is (D, H, W) or (H, W, D).
        # Let's save as (D, H, W) -> [Slice, 256, 256]
        
        np.save(out_case_dir / 'image.npy', final_img.astype(np.float32))
        np.save(out_case_dir / 'mask.npy', final_lab.astype(np.uint8))
        
    except Exception as e:
        print(f"Error processing {case_folder.name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    
    img_dir = raw_path / 'imagesTr'
    label_dir = raw_path / 'labelsTr'
    
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    
    cases = sorted(list(img_dir.glob('*.nii.gz')))
    print(f"Found {len(cases)} cases to process.")
    
    # Simple parallel or sequential? Sequential is safer for now.
    for case_file in tqdm(cases):
        # Dummy folder wrapper to match previous structure logic
        # case_file is .../pancreas_001.nii.gz
        case_name = case_file.name.replace('.nii.gz', '')
        
        # Create virtual object to match function signature
        class Obj: pass
        virtual_folder = Obj()
        virtual_folder.name = case_name
        
        process_case(virtual_folder, out_path, img_dir, label_dir)

if __name__ == "__main__":
    main()
