"""
Attention U-Net Training Script
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
from attention_unet import create_attention_unet


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
    
    train_images, train_labels = get_paths(train_folders)
    val_images, val_labels = get_paths(val_folders)
    
    return {
        'train': {'images': train_images, 'labels': train_labels},
        'validation': {'images': val_images, 'labels': val_labels}
    }


class DiceFocalComboLoss(tf.keras.losses.Loss):
    """GEMINI Loss"""
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
    
    def dice_loss(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
    
    def focal_loss(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.clip_by_value(tf.reshape(y_pred, [-1]), 1e-7, 1 - 1e-7)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = tf.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal)
    
    def call(self, y_true, y_pred):
        y_pred = tf.nn.sigmoid(y_pred)
        return self.dice_weight * self.dice_loss(y_true, y_pred) + self.focal_weight * self.focal_loss(y_true, y_pred)


class AttentionUNetTrainer:
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        print(f"\n=== Attention U-Net + GEMINI Loss ===")
        
        self.model = create_attention_unet(
            input_shape=(config.img_size_x, config.img_size_y, config.num_channels),
            n_filters=32
        )
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3, decay_steps=50000, alpha=0.01
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.loss_fn = DiceFocalComboLoss()
        self.data_pipeline = DataPipeline(config)
        
        self.history = {
            'train_loss': [], 'val_dice': [], 'val_fg_pred': [], 'learning_rate': []
        }
    
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
        y_pred = tf.nn.sigmoid(y_pred_logits)
        y_pred_binary = tf.cast(tf.squeeze(y_pred) > 0.5, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred_binary, [tf.shape(y_pred_binary)[0], -1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        sum_true = tf.reduce_sum(y_true_flat, axis=1)
        sum_pred = tf.reduce_sum(y_pred_flat, axis=1)
        
        dice = (2.0 * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
        return dice, tf.reduce_mean(sum_pred)
    
    def validate(self, val_dataset):
        all_dices = []
        total_fg_pred = 0
        num_batches = 0
        
        for images, labels in val_dataset:
            predictions = self.model(images, training=False)
            if len(labels.shape) > 3:
                labels = labels[..., -1]
            if len(predictions.shape) > 3:
                predictions = predictions[..., -1]
            
            dice, fg_pred = self.compute_dice(labels, predictions)
            all_dices.extend(dice.numpy().tolist())
            total_fg_pred += float(fg_pred)
            num_batches += 1
        
        return float(np.mean(all_dices)), total_fg_pred / max(num_batches, 1)
    
    def train(self, data_paths, num_epochs=30):
        print(f"\nStarting Attention U-Net training for {num_epochs} epochs...")
        
        train_ds = self.data_pipeline.build_labeled_dataset(
            data_paths['train']['images'], data_paths['train']['labels'],
            batch_size=self.config.batch_size, is_training=True
        )
        val_ds = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'], data_paths['validation']['labels'],
            batch_size=self.config.batch_size
        )
        
        best_dice = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_losses = []
            
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            print(f"\nEpoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}]")
            
            for images, labels in tqdm(train_ds, desc="Training"):
                loss = self.train_step(images, labels)
                epoch_losses.append(float(loss))
            
            val_dice, fg_pred = self.validate(val_ds)
            epoch_time = time.time() - start_time
            
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['val_fg_pred'].append(fg_pred)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Time: {epoch_time:.1f}s | Loss: {np.mean(epoch_losses):.4f} | Val Dice: {val_dice:.4f} | FG: {fg_pred:.0f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.model.save_weights(str(self.output_dir / 'best_attention_unet.weights.h5'))
                print(f"✓ New best! Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                print("\nEarly stopping!")
                break
            
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
                self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Attention U-Net Complete! Best Dice: {best_dice:.4f}")
        print(f"{'='*60}")
        self.plot_progress()
        self.save_history()
        return best_dice
    
    def plot_progress(self):
        if len(self.history['train_loss']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Attention U-Net Training Progress', fontweight='bold', fontsize=14)
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        
        axes[0, 1].plot(epochs, self.history['val_dice'], 'g-', linewidth=2)
        best_epoch = np.argmax(self.history['val_dice']) + 1
        best_dice = max(self.history['val_dice'])
        axes[0, 1].axhline(y=0.7349, color='orange', linestyle=':', label='GEMINI U-Net: 0.7349')
        axes[0, 1].scatter([best_epoch], [best_dice], color='red', s=100, marker='*')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].set_title(f'Val Dice (Best: {best_dice:.4f})')
        axes[0, 1].legend()
        
        axes[1, 0].plot(epochs, self.history['val_fg_pred'], 'purple', linewidth=2)
        axes[1, 0].axhline(y=534, color='orange', linestyle=':', label='GT FG')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('FG Pixels')
        axes[1, 0].set_title('Foreground Predictions')
        axes[1, 0].legend()
        
        axes[1, 1].axis('off')
        summary = f"Best Dice: {best_dice:.4f}\nBest Epoch: {best_epoch}\nImprovement vs GEMINI: {((best_dice-0.7349)/0.7349)*100:+.2f}%"
        axes[1, 1].text(0.5, 0.5, summary, ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_unet_progress.png', dpi=150)
        plt.close()
    
    def save_history(self):
        pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            **self.history
        }).to_csv(self.output_dir / 'attention_unet_history.csv', index=False)


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
    
    data_paths = get_all_data_paths(args.data_dir)
    
    exp_dir = Path(f'experiments/attention_unet_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = AttentionUNetTrainer(config, output_dir=exp_dir)
    trainer.train(data_paths, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
