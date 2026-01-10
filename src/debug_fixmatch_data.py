import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

from config import StableSSLConfig
from data_loader_tf2 import DataPipeline

def get_all_data_paths(data_dir):
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])
    
    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    # Split 221 Train 
    train_folders = all_folders[:221] 
    
    # 20% Labeled Split (~44 patients)
    n_labeled = 44 
    labeled_folders = train_folders[:n_labeled]
    
    def get_paths(folders):
        images, labels = [], []
        for folder in folders:
            img_path = folder / 'image.npy'
            mask_path = folder / 'mask.npy'
            if img_path.exists() and mask_path.exists():
                images.append(str(img_path))
                labels.append(str(mask_path))
        return images, labels
    
    return {
        'labeled': dict(zip(['images', 'labels'], get_paths(labeled_folders))),
    }

def main():
    print("DEBUG: Identifying Data Pipeline Issues")
    config = StableSSLConfig()
    config.batch_size = 4
    
    paths = get_all_data_paths(Path('preprocessed_v2'))
    print(f"Labeled Images Found: {len(paths['labeled']['images'])}")
    print(f"Labeled Labels Found: {len(paths['labeled']['labels'])}")
    
    pipeline = DataPipeline(config)
    ds_l = pipeline.build_labeled_dataset(
        paths['labeled']['images'], paths['labeled']['labels'],
        batch_size=4
    )
    
    print("\nIterating 1 batch...")
    for x, y in ds_l.take(1):
        print(f"X shape: {x.shape}, Max: {np.max(x)}, Min: {np.min(x)}")
        print(f"Y shape: {y.shape}, Max: {np.max(y)}, Min: {np.min(y)}, Unique: {np.unique(y)}")
        
        if np.max(y) == 0:
            print("CRITICAL: Label batch is all zeros!")
        else:
            print("Labels look okay (contains 1s).")

if __name__ == "__main__":
    main()
