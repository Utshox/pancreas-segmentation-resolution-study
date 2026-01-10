"""
Phase 5: High-Resolution Baseline (512x512)
Data: V4 (512x512, Strict HU) - 100% Labeled
Model: ResNet50-UNet (ImageNet Weights)
Batch Size: 8 (Memory Constraint)
Optimizations: Mixed Precision (Robusted)
"""
import sys
import os
import numpy as np
import time
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow.keras import mixed_precision 
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Segmentation Models (SM) needs to be installed in venv
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')
from config import StableSSLConfig
from train_ssl_tf2n import setup_gpu
from data_loader_tf2 import DataPipeline

# Enable Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Computed precision: {policy.compute_dtype}")
print(f"Variable precision: {policy.variable_dtype}")

def get_data_paths(data_dir):
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])
    
    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    train_folders = all_folders[:221] 
    val_folders = all_folders[221:]
    
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
        'train': dict(zip(['images', 'labels'], get_paths(train_folders))),
        'validation': dict(zip(['images', 'labels'], get_paths(val_folders)))
    }

class TransferTrainer:
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        # Build Model
        # Using linear activation and handling sigmoid in loss is cleaner for MP, but let's stick to standard
        self.model = sm.Unet('resnet50', classes=1, activation='sigmoid', encoder_weights='imagenet', input_shape=(None, None, 3))
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-4, 100000) 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # WRAP OPTIMIZER - Try explicit import to be safe
        try:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
            print("Wrapped optimizer with mixed_precision.LossScaleOptimizer")
        except:
            print("Warning: Could not wrap optimizer. Mixed Precision might be unstable.")

        self.dice_loss = sm.losses.DiceLoss()
        self.focal_loss = sm.losses.BinaryFocalLoss()
        self.data_pipeline = DataPipeline(config)
        self.history = {'train_loss': [], 'val_dice': []}
        
    def gemini_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return 0.5 * self.dice_loss(y_true, y_pred) + 0.5 * self.focal_loss(y_true, y_pred)
    
    def dice_coef(self, y_true, y_pred):
        smooth = 1e-6
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss = self.gemini_loss(y, pred)
            
            # ROBUST SCALING LOGIC
            scaled_loss = loss
            if hasattr(self.optimizer, 'get_scaled_loss'):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            elif hasattr(self.optimizer, 'scale_loss'):
                scaled_loss = self.optimizer.scale_loss(loss)
            
        scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # ROBUST UNSCALING LOGIC
        grads = scaled_grads
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def validate(self, val_ds):
        dices = []
        for x, y in val_ds:
            pred = self.model(x, training=False)
            d = self.dice_coef(y, pred) 
            dices.append(float(d))
        return np.mean(dices)

    def train(self, data_paths, num_epochs):
        print(f"Starting High-Res Training (Robust MP) for {num_epochs} epochs...")
        
        train_ds_raw = self.data_pipeline.build_labeled_dataset(
            data_paths['train']['images'], data_paths['train']['labels'],
            batch_size=self.config.batch_size
        )
        val_ds_raw = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'], data_paths['validation']['labels'],
            batch_size=self.config.batch_size
        )

        preprocess_input = sm.get_preprocessing('resnet50')

        def preprocess_data(x, y):
            x = tf.repeat(x, 3, axis=-1)
            x = x * 255.0
            x = preprocess_input(x)
            return x, y
        
        train_ds = train_ds_raw.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds_raw.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        best_dice = 0.0
        patience_counter = 0
        patience_limit = 15
        
        for epoch in range(num_epochs):
            losses = []
            
            for x, y in tqdm(train_ds, desc=f"Epoch {epoch+1}"):
                l = self.train_step(x, y)
                losses.append(float(l))
                
            val_dice = self.validate(val_ds)
            print(f"Epoch {epoch+1}: Val Dice: {val_dice:.4f} | Loss: {np.mean(losses):.4f}")
            
            self.history['train_loss'].append(np.mean(losses))
            self.history['val_dice'].append(val_dice)
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.model.save_weights(self.output_dir / 'best_model.weights.h5')
                patience_counter = 0
                print(f" -> New Best Model Saved! ({best_dice:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                 print("Early stopping triggered.")
                 break
            
            if (epoch+1) % 5 == 0:
                self.save_history()
                
        self.plot_progress()
        self.save_history()
        
    def plot_progress(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['val_dice'], label='Val Dice')
        plt.plot(self.history['train_loss'], label='Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'high_res_progress.png')
        plt.close()

    def save_history(self):
        pd.DataFrame(self.history).to_csv(self.output_dir / 'high_res_history.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    setup_gpu()
    
    config = StableSSLConfig()
    config.batch_size = args.batch_size
    
    exp_dir = Path(f'experiments/high_res_512_mp_fixed_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    paths = get_data_paths(Path('preprocessed_v4_512'))
    
    trainer = TransferTrainer(config, output_dir=exp_dir)
    trainer.train(paths, args.num_epochs)

if __name__ == "__main__":
    main()
