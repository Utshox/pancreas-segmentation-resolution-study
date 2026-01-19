
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def create_visualization(image_path, label_path, pred_path, output_path, slice_idx=None):
    # Load NIfTI files
    img_obj = nib.load(image_path)
    lbl_obj = nib.load(label_path)
    pred_obj = nib.load(pred_path)
    
    img = img_obj.get_fdata()
    lbl = lbl_obj.get_fdata()
    pred = pred_obj.get_fdata()
    
    # Find slice with max tumor area if not specified
    if slice_idx is None:
        tumor_counts = np.sum(lbl, axis=(0, 1))
        # If no tumor in GT, try prediction
        if np.sum(tumor_counts) == 0:
             tumor_counts = np.sum(pred, axis=(0, 1))
        
        if np.sum(tumor_counts) > 0:
            slice_idx = np.argmax(tumor_counts)
        else:
            slice_idx = img.shape[2] // 2 # Fallback
    
    # Extract slices
    img_slice = np.rot90(img[:, :, slice_idx])
    lbl_slice = np.rot90(lbl[:, :, slice_idx])
    pred_slice = np.rot90(pred[:, :, slice_idx])
    
    # Normalize image for display
    img_slice = normalize(img_slice)
    
    # Create Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. CT Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f'CT Scan (Slice {slice_idx})', fontsize=14)
    axes[0].axis('off')
    
    # 2. Ground Truth Overlay
    axes[1].imshow(img_slice, cmap='gray')
    # Create RGBA overlay (Green for GT)
    gt_overlay = np.zeros(lbl_slice.shape + (4,))
    gt_overlay[lbl_slice == 1] = [0, 1, 0, 0.4] # Green, alpha 0.4
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth (Green)', fontsize=14)
    axes[1].axis('off')
    
    # 3. Prediction Overlay
    axes[2].imshow(img_slice, cmap='gray')
    # Create RGBA overlay (Red for Pred)
    pred_overlay = np.zeros(pred_slice.shape + (4,))
    pred_overlay[pred_slice == 1] = [1, 0, 0, 0.4] # Red, alpha 0.4
    
    # Calculate Dice for this slice
    intersection = np.sum(lbl_slice * pred_slice)
    dice = 2. * intersection / (np.sum(lbl_slice) + np.sum(pred_slice) + 1e-8)
    
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f'Prediction (Red) - Dice: {dice:.2f}', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f'Case: {os.path.basename(image_path)}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='Path to save PNG')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predictions')
    parser.add_argument('--case_id', type=str, default='pancreas_001.nii.gz', help='Case ID')
    args = parser.parse_args()

    # Base data directory
    base_dir = os.path.expanduser('~/ish/data_val')
    
    case_id = args.case_id
    
    image_path = os.path.join(base_dir, 'imagesTr', case_id)
    label_path = os.path.join(base_dir, 'labelsTr', case_id)
    pred_path = os.path.join(args.pred_dir, case_id)
    
    if os.path.exists(pred_path):
        create_visualization(image_path, label_path, pred_path, args.output_path)
    else:
        print(f"Prediction file not found: {pred_path}")

if __name__ == "__main__":
    main()
