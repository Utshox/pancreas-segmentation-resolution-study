"""
GEMINI Version: Supervised Training with Dice + Focal Loss Combo
0.5 × DiceLoss + 0.5 × FocalLoss
Good all-rounder that fixes the "all-background" trap
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
from models_tf2 import PancreasSeg


def get_all_data_paths(data_dir):
    """Get all 281 patients for supervised training (221 train, 60 val)"""
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])
    
    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    
    train_folders = all_folders[:221]
    val_folders = all_folders[221:]
    
    print(f"\nSupervised Split:")
    print(f"  Train: {len(train_folders)} | Val: {len(val_folders)}")
    
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
    """
    Dice + Focal Loss Combo
    0.5 × DiceLoss + 0.5 × FocalLoss
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma  # Focal parameter
        self.alpha = alpha  # Class balance for focal
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
        y_pred = tf.reshape(y_pred, [-1])
        
        # Clip to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Focal loss formula
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = tf.where(y_true == 1, self.alpha, 1 - self.alpha)
        
        focal = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        
        return tf.reduce_mean(focal)
    
    def call(self, y_true, y_pred):
        # Apply sigmoid if logits
        y_pred = tf.nn.sigmoid(y_pred)
        
        dice = self.dice_loss(y_true, y_pred)
        focal = self.focal_loss(y_true, y_pred)
        
        return self.dice_weight * dice + self.focal_weight * focal


class SupervisedTrainerGemini:
    """Supervised training with Dice + Focal Combo Loss (GEMINI version)"""
    
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        print(f"\n=== GEMINI: Supervised + Dice+Focal Combo ===")
        print(f"  0.5 × DiceLoss + 0.5 × FocalLoss")
        print(f"  Focal γ: 2.0, α: 0.25")
        
        self.model = PancreasSeg(config)
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.model(dummy_input)
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3, decay_steps=50000, alpha=0.01
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # DICE + FOCAL COMBO LOSS
        self.loss_fn = DiceFocalComboLoss(dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25)
        
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
        
        for batch in val_dataset:
            images, labels = batch
            predictions = self.model(images, training=False)
            
            if len(labels.shape) > 3:
                labels = labels[..., -1]
            if len(predictions.shape) > 3:
                predictions = predictions[..., -1]
            
            dice, fg_pred = self.compute_dice(labels, predictions)
            all_dices.extend(dice.numpy().tolist())
            total_fg_pred += float(fg_pred)
            num_batches += 1
        
        mean_dice = np.mean(all_dices) if all_dices else 0.0
        avg_fg_pred = total_fg_pred / max(num_batches, 1)
        
        return float(mean_dice), avg_fg_pred
    
    def train(self, data_paths, num_epochs=30):
        print(f"\nStarting GEMINI training for {num_epochs} epochs...")
        
        train_ds = self.data_pipeline.build_labeled_dataset(
            data_paths['train']['images'],
            data_paths['train']['labels'],
            batch_size=self.config.batch_size,
            is_training=True
        )
        
        val_ds = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size
        )
        
        best_dice = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_losses = []
            
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            print(f"\nEpoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}]")
            progress = tqdm(train_ds, desc="Training")
            
            for images, labels in progress:
                loss = self.train_step(images, labels)
                epoch_losses.append(float(loss))
                progress.set_postfix({'loss': f'{float(loss):.4f}'})
            
            val_dice, fg_pred = self.validate(val_ds)
            epoch_time = time.time() - start_time
            
            avg_loss = np.mean(epoch_losses)
            
            self.history['train_loss'].append(avg_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_fg_pred'].append(fg_pred)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Time: {epoch_time:.1f}s | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | FG Pred: {fg_pred:.0f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_gemini')
                print(f"✓ New best! Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                print("\nEarly stopping triggered!")
                break
            
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
                self.save_history_csv()
        
        print(f"\n{'='*60}")
        print(f"GEMINI Training Complete! Best Dice: {best_dice:.4f}")
        print(f"{'='*60}")
        self.plot_progress()
        self.save_history_csv()
        return best_dice
    
    def save_checkpoint(self, name):
        checkpoint_dir = Path('supervised_gemini/checkpoints') / time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(checkpoint_dir / f'{name}.weights.h5'))
        print(f"Saved to {checkpoint_dir}")
    
    def plot_progress(self):
        if len(self.history['train_loss']) < 2:
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('GEMINI: Supervised + Dice+Focal Combo', fontweight='bold', fontsize=16, y=0.98)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dice+Focal Loss')
        ax1.set_title('Training Loss', fontweight='bold')
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(epochs, self.history['val_dice'], 'g-', linewidth=2)
        best_epoch = np.argmax(self.history['val_dice']) + 1
        best_dice = max(self.history['val_dice'])
        ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7)
        ax2.scatter([best_epoch], [best_dice], color='red', s=100, marker='*')
        ax2.axhline(y=0.7028, color='orange', linestyle=':', label='BCE Baseline (0.7028)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.set_title(f'Validation Dice (Best: {best_dice:.4f})', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(epochs, self.history['val_fg_pred'], 'purple', linewidth=2)
        ax3.axhline(y=534, color='orange', linestyle=':', label='Ground Truth (~534)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Predicted FG Pixels')
        ax3.set_title('Foreground Prediction Volume', fontweight='bold')
        ax3.legend()
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        improvement = ((best_dice - 0.7028) / 0.7028) * 100
        summary = f"""
╔══════════════════════════════════════════════════╗
║          GEMINI: DICE + FOCAL COMBO              ║
╠══════════════════════════════════════════════════╣
║  Loss: 0.5×Dice + 0.5×Focal                      ║
║  Focal γ: 2.0                                    ║
║  Focal α: 0.25                                   ║
╠══════════════════════════════════════════════════╣
║  BCE Baseline: 0.7028                            ║
║  GEMINI Best:  {best_dice:.4f}                            ║
║  Improvement:  {improvement:+.2f}%                           ║
╠══════════════════════════════════════════════════╣
║  Best Epoch: {best_epoch}                                   ║
╚══════════════════════════════════════════════════╝
"""
        ax4.text(0.5, 0.5, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / 'gemini_progress.png', dpi=150)
        plt.savefig(self.output_dir / 'gemini_progress.pdf')
        plt.close()
        print(f"Plots saved to {self.output_dir}")
    
    def save_history_csv(self):
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_dice': self.history['val_dice'],
            'val_fg_pred': self.history['val_fg_pred'],
            'learning_rate': self.history['learning_rate']
        })
        df.to_csv(self.output_dir / 'gemini_history.csv', index=False)


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
    
    exp_dir = Path(f'experiments/gemini_dice_focal_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = SupervisedTrainerGemini(config, output_dir=exp_dir)
    
    best_dice = trainer.train(data_paths, num_epochs=args.num_epochs)
    print(f"\nGEMINI Results in: {exp_dir}")
    print(f"Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
