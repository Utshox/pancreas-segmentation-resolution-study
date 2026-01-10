import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import multiprocessing

# Config
TARGET_SHAPE = (512, 512) # High Res
HU_LOWER = -125
HU_UPPER = 275
# CORRECTED PATH: Data is in ~/ish/imagesTr
RAW_DATA_DIR = Path.home() / 'ish/imagesTr'
LABEL_DIR = Path.home() / 'ish/labelsTr'
OUTPUT_DIR = Path('preprocessed_v4_512')
OUTPUT_DIR.mkdir(exist_ok=True)

def process_case(case_name):
    # Skip if done
    case_out_dir = OUTPUT_DIR / case_name.replace('.nii.gz', '')
    if case_out_dir.exists():
        return
        
    try:
        # Load
        if not (RAW_DATA_DIR / case_name).exists():
            return
            
        img_nii = nib.load(RAW_DATA_DIR / case_name)
        img_data = img_nii.get_fdata()
        
        lbl_path = LABEL_DIR / case_name
        if not lbl_path.exists():
            return
        lbl_data = nib.load(lbl_path).get_fdata()

        # Windowing (Strict)
        img_data = np.clip(img_data, HU_LOWER, HU_UPPER)
        img_data = (img_data - HU_LOWER) / (HU_UPPER - HU_LOWER) # [0, 1]

        # Resize
        z_depth = img_data.shape[2]
        img_resized = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], z_depth), dtype=np.float32)
        lbl_resized = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], z_depth), dtype=np.uint8)
        
        valid_slices = False
        
        for i in range(z_depth):
            img_slice = img_data[..., i]
            lbl_slice = lbl_data[..., i]
            
            if np.sum(lbl_slice) > 0:
                valid_slices = True
            
            img_resized[..., i] = resize(img_slice, TARGET_SHAPE, order=1, preserve_range=True, anti_aliasing=True)
            lbl_resized[..., i] = resize(lbl_slice, TARGET_SHAPE, order=0, preserve_range=True, anti_aliasing=False) 

        # Save
        case_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_out_dir / 'image.npy', img_resized)
        np.save(case_out_dir / 'mask.npy', lbl_resized)
        
    except Exception as e:
        print(f"Error {case_name}: {e}")

def main():
    if not RAW_DATA_DIR.exists():
        print(f"ERROR: {RAW_DATA_DIR} does not exist!")
        return

    cases = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.nii.gz') and not f.startswith('.')]
    print(f"Found {len(cases)} cases in {RAW_DATA_DIR}. Processing to {TARGET_SHAPE}...")
    
    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_case, cases), total=len(cases)))
    
    print("Preprocessing V4 (512x512) Completed.")

if __name__ == '__main__':
    main()
