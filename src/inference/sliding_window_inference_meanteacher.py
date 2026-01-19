
import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

# --- U-Net (Must match training exactly) ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))

    # Encoding
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoding
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Mean Teacher Wrapper (Must match training exactly for weight loading) ---
class MeanTeacherTrainer(models.Model):
    def __init__(self, student_model, teacher_model, alpha=0.999, lambda_u=10.0):
        super(MeanTeacherTrainer, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        # We don't need the rest for inference, but __init__ is called on load
        self.alpha = alpha
        self.lambda_u = lambda_u
    
    def call(self, inputs):
        # Call BOTH to ensure both are built and weights can be loaded
        _ = self.student(inputs)
        return self.teacher(inputs)

def normalize(volume):
    # MATCH PREPROCESSING LOGIC EXACTLY (preprocess_v5_patches.py)
    HU_MIN, HU_MAX = -125, 275
    volume = np.clip(volume, HU_MIN, HU_MAX)
    return (volume - HU_MIN) / (HU_MAX - HU_MIN)

def sliding_window_inference(model, volume, patch_size=(256, 256), stride=128):
    h, w, d = volume.shape
    prediction = np.zeros(volume.shape, dtype=np.float32)
    count = np.zeros(volume.shape, dtype=np.float32)

    # Simple 2D sliding window on each slice
    for z in range(d):
        slice_img = volume[:, :, z]
        
        # Determine number of patches
        h_steps = int(np.ceil((h - patch_size[0]) / stride)) + 1
        w_steps = int(np.ceil((w - patch_size[1]) / stride)) + 1
        
        for i in range(h_steps):
            for j in range(w_steps):
                y_start = min(i * stride, h - patch_size[0])
                x_start = min(j * stride, w - patch_size[1])
                y_end = y_start + patch_size[0]
                x_end = x_start + patch_size[1]
                
                patch = slice_img[y_start:y_end, x_start:x_end]
                patch_input = np.expand_dims(patch, axis=[0, -1]) # (1, 256, 256, 1)
                
                pred_patch = model.predict(patch_input, verbose=0)[0, :, :, 0]
                
                prediction[y_start:y_end, x_start:x_end, z] += pred_patch
                count[y_start:y_end, x_start:x_end, z] += 1.0

    return prediction / np.maximum(count, 1e-8)

def compute_dice(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw NIfTI files (imagesTr/labelsTr)')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # 1. Create Teacher Model (no wrapper needed!)
    teacher = get_unet()

    print(f"Loading weights from {args.model_path}...")
    try:
        # Load Clean Weights (matches standard model structure)
        teacher.load_weights(args.model_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 2. Use teacher for inference
    inference_model = teacher

    # 4. Run Inference
    images_dir = os.path.join(args.data_dir, 'imagesTr')
    labels_dir = os.path.join(args.data_dir, 'labelsTr')
    os.makedirs(args.output_dir, exist_ok=True)

    test_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    # Taking last 4 files as test set (consistent with previous experiments)
    test_files = test_files[-4:] 
    
    dice_scores = []

    print(f"Running inference on {len(test_files)} test cases...")

    for f in tqdm(test_files):
        img_path = os.path.join(images_dir, f)
        lbl_path = os.path.join(labels_dir, f)
        
        img_obj = nib.load(img_path)
        lbl_obj = nib.load(lbl_path)
        
        img = img_obj.get_fdata().astype(np.float32)
        lbl = lbl_obj.get_fdata().astype(np.float32)
        
        # Preprocessing: Normalize
        img = normalize(img)
        
        # Inference
        pred_prob = sliding_window_inference(inference_model, img)
        pred_mask = (pred_prob > 0.5).astype(np.float32)
        
        # Save Prediction
        save_path = os.path.join(args.output_dir, f)
        nib.save(nib.Nifti1Image(pred_mask, img_obj.affine), save_path)
        
        # Metric
        dice = compute_dice(lbl, pred_mask)
        dice_scores.append(dice)
        print(f"Case {f}: Dice = {dice:.4f}")

    print("-" * 30)
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
