"""
Dual-Encoder Fourier Training Script
Architecture experiment with GEMINI loss
"""
import sys
import os
import numpy as np
import time
from pathlib import Path
import argparse

sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from config import StableSSLConfig
from train_ssl_tf2n import setup_gpu
from data_loader_tf2 import DataPipeline
from dual_encoder_fourier import create_dual_encoder_fourier


def get_all_data_paths(data_dir):
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])
    
    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    train_folders = all_folders[:221]
    val_folders = all_folders[221:]
    
    print(f"\nSupervised Split: Train: {len(train_folders)} | Val: {len(val_folders)}")
    
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


class DiceFocalComboLoss(tf.keras.losses.Loss):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight, self.focal_weight = dice_weight, focal_weight
        self.gamma, self.alpha, self.smooth = gamma, alpha, smooth
    
    def call(self, y_true, y_pred):
        y_pred = tf.nn.sigmoid(y_pred)
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Dice
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2.0 * intersection + self.smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
        dice_loss = 1.0 - dice
        
        # Focal
        y_pred_f = tf.clip_by_value(y_pred_f, 1e-7, 1 - 1e-7)
        pt = tf.where(y_true_f == 1, y_pred_f, 1 - y_pred_f)
        alpha_t = tf.where(y_true_f == 1, self.alpha, 1 - self.alpha)
        focal_loss = tf.reduce_mean(-alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt))
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class DualEncoderTrainer:
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        print(f"\n=== Dual-Encoder (CNN + Fourier) + GEMINI Loss ===")
        
        self.model = create_dual_encoder_fourier(
            input_shape=(config.img_size_x, config.img_size_y, config.num_channels),
            n_filters=32
        )
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 50000, alpha=0.01)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.loss_fn = DiceFocalComboLoss()
        self.data_pipeline = DataPipeline(config)
        
        self.history = {'train_loss': [], 'val_dice': [], 'val_fg_pred': [], 'learning_rate': []}
    
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def compute_dice(self, y_true, y_pred_logits):
        y_true = tf.cast(tf.squeeze(y_true), tf.float32)
        y_pred = tf.cast(tf.nn.sigmoid(y_pred_logits) > 0.5, tf.float32)
        y_pred = tf.squeeze(y_pred)
        
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice, tf.reduce_mean(tf.reduce_sum(y_pred_flat, axis=1))
    
    def validate(self, val_dataset):
        all_dices, total_fg = [], 0
        for images, labels in val_dataset:
            predictions = self.model(images, training=False)
            if len(labels.shape) > 3: labels = labels[..., -1]
            if len(predictions.shape) > 3: predictions = predictions[..., -1]
            dice, fg = self.compute_dice(labels, predictions)
            all_dices.extend(dice.numpy().tolist())
            total_fg += float(fg)
        return float(np.mean(all_dices)), total_fg / max(len(all_dices)/8, 1)
    
    def train(self, data_paths, num_epochs=30):
        print(f"\nStarting Dual-Encoder training for {num_epochs} epochs...")
        
        train_ds = self.data_pipeline.build_labeled_dataset(
            data_paths['train']['images'], data_paths['train']['labels'],
            batch_size=self.config.batch_size, is_training=True
        )
        val_ds = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'], data_paths['validation']['labels'],
            batch_size=self.config.batch_size
        )
        
        best_dice, patience = 0, 0
        
        for epoch in range(num_epochs):
            start = time.time()
            losses = [float(self.train_step(img, lbl)) for img, lbl in tqdm(train_ds, desc=f"Epoch {epoch+1}")]
            val_dice, fg = self.validate(val_ds)
            
            lr = float(self.lr_schedule(self.optimizer.iterations))
            self.history['train_loss'].append(np.mean(losses))
            self.history['val_dice'].append(val_dice)
            self.history['val_fg_pred'].append(fg)
            self.history['learning_rate'].append(lr)
            
            print(f"Time: {time.time()-start:.1f}s | Loss: {np.mean(losses):.4f} | Val Dice: {val_dice:.4f} | FG: {fg:.0f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.model.save_weights(str(self.output_dir / 'best_dual_encoder.weights.h5'))
                print(f"✓ New best! Dice: {best_dice:.4f}")
                patience = 0
            else:
                patience += 1
            
            if patience >= 10:
                print("\nEarly stopping!")
                break
            
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
                self.save_history()
        
        print(f"\n{'='*60}\nDual-Encoder Complete! Best Dice: {best_dice:.4f}\n{'='*60}")
        self.plot_progress()
        self.save_history()
        return best_dice
    
    def plot_progress(self):
        if len(self.history['train_loss']) < 2: return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dual-Encoder (CNN + Fourier) Training', fontweight='bold')
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-')
        axes[0, 0].set_title('Training Loss')
        
        axes[0, 1].plot(epochs, self.history['val_dice'], 'g-')
        axes[0, 1].axhline(y=0.7349, color='orange', linestyle=':', label='GEMINI U-Net')
        axes[0, 1].scatter([np.argmax(self.history['val_dice'])+1], [max(self.history['val_dice'])], c='red', s=100, marker='*')
        axes[0, 1].set_title(f'Val Dice (Best: {max(self.history["val_dice"]):.4f})')
        axes[0, 1].legend()
        
        axes[1, 0].plot(epochs, self.history['val_fg_pred'], 'purple')
        axes[1, 0].axhline(y=534, color='orange', linestyle=':')
        axes[1, 0].set_title('FG Predictions')
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, f"Best: {max(self.history['val_dice']):.4f}\nvs GEMINI: {((max(self.history['val_dice'])-0.7349)/0.7349)*100:+.2f}%",
                       ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dual_encoder_progress.png', dpi=150)
        plt.close()
    
    def save_history(self):
        pd.DataFrame({'epoch': range(1, len(self.history['train_loss'])+1), **self.history}).to_csv(
            self.output_dir / 'dual_encoder_history.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('preprocessed_v2'))
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    setup_gpu()
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    
    exp_dir = Path(f'experiments/dual_encoder_fourier_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = DualEncoderTrainer(config, output_dir=exp_dir)
    trainer.train(get_all_data_paths(args.data_dir), num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
