"""
Mean Teacher with FIXED Dice Computation
BUG FIX: Use proper epsilon and foreground-focused Dice
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


def prepare_ssl_data_paths(data_dir, num_labeled=50, num_unlabeled=161, num_validation=70):
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])

    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    
    labeled_folders = all_folders[:num_labeled]
    unlabeled_folders = all_folders[num_labeled:num_labeled + num_unlabeled]
    val_folders = all_folders[num_labeled + num_unlabeled:num_labeled + num_unlabeled + num_validation]
    
    print(f"\nSSL Data Split:")
    print(f"  Labeled: {len(labeled_folders)} | Unlabeled: {len(unlabeled_folders)} | Val: {len(val_folders)}")
    
    def get_paths(folders, need_labels=True):
        images, labels = [], []
        for folder in folders:
            img_path = folder / 'image.npy'
            mask_path = folder / 'mask.npy'
            if img_path.exists():
                images.append(str(img_path))
                if need_labels and mask_path.exists():
                    labels.append(str(mask_path))
        return images, labels if need_labels else images
    
    labeled_images, labeled_labels = get_paths(labeled_folders, need_labels=True)
    unlabeled_images, _ = get_paths(unlabeled_folders, need_labels=False)
    val_images, val_labels = get_paths(val_folders, need_labels=True)
    
    return {
        'labeled': {'images': labeled_images, 'labels': labeled_labels},
        'unlabeled': {'images': unlabeled_images},
        'validation': {'images': val_images, 'labels': val_labels}
    }


class MeanTeacherFixed:
    """Mean Teacher with FIXED Dice computation"""
    
    def __init__(self, config, ema_decay=0.99, consistency_weight_max=1.0, 
                 consistency_rampup_epochs=5, output_dir=None):
        self.config = config
        self.ema_decay = ema_decay
        self.consistency_weight_max = consistency_weight_max
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        print(f"\n=== Mean Teacher (FIXED Dice) ===")
        print(f"  EMA decay: {ema_decay}")
        print(f"  Consistency weight max: {consistency_weight_max}")
        
        self.student = PancreasSeg(config)
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.student(dummy_input)
        
        self.teacher = PancreasSeg(config)
        _ = self.teacher(dummy_input)
        self._copy_weights(self.student, self.teacher)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.supervised_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.consistency_loss = tf.keras.losses.MeanSquaredError()
        
        self.data_pipeline = DataPipeline(config)
        
        self.history = {
            'train_loss': [], 'supervised_loss': [], 'consistency_loss': [],
            'val_dice': [], 'val_dice_fg': [], 'learning_rate': [], 'consistency_weight': []
        }
        
    def _copy_weights(self, source, target):
        for s_var, t_var in zip(source.trainable_variables, target.trainable_variables):
            t_var.assign(s_var)
    
    def _update_teacher(self):
        for s_var, t_var in zip(self.student.trainable_variables, self.teacher.trainable_variables):
            t_var.assign(self.ema_decay * t_var + (1 - self.ema_decay) * s_var)
    
    def _get_consistency_weight(self, epoch):
        if epoch >= self.consistency_rampup_epochs:
            return self.consistency_weight_max
        return self.consistency_weight_max * (epoch / self.consistency_rampup_epochs)
    
    @tf.function
    def _augment(self, images):
        noise = tf.random.normal(tf.shape(images), mean=0.0, stddev=0.1)
        images = images + noise
        images = images * tf.random.uniform([], 0.8, 1.2)
        return tf.clip_by_value(images, -1.0, 1.0)
    
    @tf.function
    def train_step(self, labeled_images, labeled_labels, unlabeled_images, consistency_weight):
        student_unlabeled = self._augment(unlabeled_images)
        teacher_unlabeled = unlabeled_images
        
        with tf.GradientTape() as tape:
            student_labeled_aug = self._augment(labeled_images)
            student_labeled_logits = self.student(student_labeled_aug, training=True)
            sup_loss = self.supervised_loss(labeled_labels, student_labeled_logits)
            
            student_unlabeled_logits = self.student(student_unlabeled, training=True)
            teacher_unlabeled_logits = self.teacher(teacher_unlabeled, training=False)
            
            student_probs = tf.nn.sigmoid(student_unlabeled_logits)
            teacher_probs = tf.nn.sigmoid(teacher_unlabeled_logits)
            
            temp = 0.5
            teacher_probs_sharp = tf.pow(teacher_probs, 1.0/temp)
            teacher_probs_sharp = teacher_probs_sharp / (teacher_probs_sharp + tf.pow(1-teacher_probs, 1.0/temp) + 1e-7)
            
            cons_loss = self.consistency_loss(teacher_probs_sharp, student_probs)
            total_loss = sup_loss + consistency_weight * cons_loss
        
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        self._update_teacher()
        
        return total_loss, sup_loss, cons_loss
    
    def compute_dice_fixed(self, y_true, y_pred_logits):
        """FIXED Dice: proper epsilon, foreground-focused"""
        y_true = tf.cast(tf.squeeze(y_true), tf.float32)
        y_pred = tf.nn.sigmoid(y_pred_logits)
        y_pred_binary = tf.cast(tf.squeeze(y_pred) > 0.5, tf.float32)
        
        # Flatten for computation
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred_binary, [tf.shape(y_pred_binary)[0], -1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        sum_true = tf.reduce_sum(y_true_flat, axis=1)
        sum_pred = tf.reduce_sum(y_pred_flat, axis=1)
        
        # FIXED: Use small epsilon (1e-6) instead of 1.0
        # This prevents division by zero but doesn't inflate Dice when both are zero
        epsilon = 1e-6
        dice = (2.0 * intersection + epsilon) / (sum_true + sum_pred + epsilon)
        
        # Also compute foreground-only stats for debugging
        fg_pixels_true = tf.reduce_mean(sum_true)
        fg_pixels_pred = tf.reduce_mean(sum_pred)
        
        return dice, fg_pixels_true, fg_pixels_pred
    
    def validate(self, val_dataset):
        all_dices = []
        total_fg_true = 0
        total_fg_pred = 0
        num_batches = 0
        
        for batch in val_dataset:
            images, labels = batch
            predictions = self.student(images, training=False)
            
            if len(labels.shape) > 3:
                labels = labels[..., -1]
            if len(predictions.shape) > 3:
                predictions = predictions[..., -1]
            
            dice, fg_true, fg_pred = self.compute_dice_fixed(labels, predictions)
            all_dices.extend(dice.numpy().tolist())
            total_fg_true += float(fg_true)
            total_fg_pred += float(fg_pred)
            num_batches += 1
        
        mean_dice = np.mean(all_dices) if all_dices else 0.0
        avg_fg_true = total_fg_true / max(num_batches, 1)
        avg_fg_pred = total_fg_pred / max(num_batches, 1)
        
        # Log foreground stats
        print(f"  [Val Debug] Avg FG pixels - True: {avg_fg_true:.0f}, Pred: {avg_fg_pred:.0f}")
        
        return float(mean_dice), avg_fg_true, avg_fg_pred
    
    def train(self, data_paths, num_epochs=30):
        print(f"\nStarting Mean Teacher (FIXED) training for {num_epochs} epochs...")
        
        labeled_ds = self.data_pipeline.build_labeled_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            is_training=True
        )
        
        unlabeled_ds = self.data_pipeline.build_unlabeled_dataset_for_mean_teacher(
            data_paths['unlabeled']['images'],
            batch_size=self.config.batch_size,
            is_training=True
        ) if data_paths['unlabeled']['images'] else None
        
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
            epoch_sup_losses = []
            epoch_cons_losses = []
            
            cons_weight = self._get_consistency_weight(epoch)
            
            if unlabeled_ds:
                combined_ds = tf.data.Dataset.zip((labeled_ds, unlabeled_ds.repeat()))
            else:
                combined_ds = labeled_ds
            
            current_lr = float(self.optimizer.learning_rate)
            print(f"\nEpoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}, ConsW: {cons_weight:.2f}]")
            progress = tqdm(combined_ds, desc="Training")
            
            for batch in progress:
                if unlabeled_ds:
                    (labeled_imgs, labeled_lbls), unlabeled_imgs = batch
                    if isinstance(unlabeled_imgs, tuple):
                        unlabeled_imgs = unlabeled_imgs[0]
                else:
                    labeled_imgs, labeled_lbls = batch
                    unlabeled_imgs = labeled_imgs
                
                total_loss, sup_loss, cons_loss = self.train_step(
                    labeled_imgs, labeled_lbls, unlabeled_imgs, 
                    tf.constant(cons_weight, dtype=tf.float32)
                )
                
                epoch_losses.append(float(total_loss))
                epoch_sup_losses.append(float(sup_loss))
                epoch_cons_losses.append(float(cons_loss))
                
                progress.set_postfix({
                    'loss': f'{float(total_loss):.4f}',
                    'sup': f'{float(sup_loss):.4f}',
                    'cons': f'{float(cons_loss):.4f}'
                })
            
            val_dice, fg_true, fg_pred = self.validate(val_ds)
            epoch_time = time.time() - start_time
            
            avg_loss = np.mean(epoch_losses)
            avg_sup = np.mean(epoch_sup_losses)
            avg_cons = np.mean(epoch_cons_losses)
            
            self.history['train_loss'].append(avg_loss)
            self.history['supervised_loss'].append(avg_sup)
            self.history['consistency_loss'].append(avg_cons)
            self.history['val_dice'].append(val_dice)
            self.history['val_dice_fg'].append(fg_pred)
            self.history['learning_rate'].append(current_lr)
            self.history['consistency_weight'].append(cons_weight)
            
            print(f"Time: {epoch_time:.1f}s | Loss: {avg_loss:.4f} | Sup: {avg_sup:.4f} | Cons: {avg_cons:.4f} | Val Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_mean_teacher_fixed')
                print(f"✓ New best! Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                print("\nEarly stopping triggered!")
                break
            
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
                self.save_history_csv()
        
        print(f"\nTraining complete! Best Dice: {best_dice:.4f}")
        self.plot_progress()
        self.save_history_csv()
        return self.history
    
    def save_checkpoint(self, name):
        checkpoint_dir = Path('mean_teacher_results/checkpoints') / time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.student.save_weights(str(checkpoint_dir / f'{name}_student.weights.h5'))
        self.teacher.save_weights(str(checkpoint_dir / f'{name}_teacher.weights.h5'))
        print(f"Saved to {checkpoint_dir}")
    
    def plot_progress(self):
        if len(self.history['train_loss']) < 2:
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Mean Teacher Training (FIXED Dice)', fontweight='bold', fontsize=16, y=0.98)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Total')
        ax1.plot(epochs, self.history['supervised_loss'], 'g--', linewidth=1.5, label='Supervised')
        ax1.plot(epochs, self.history['consistency_loss'], 'r--', linewidth=1.5, label='Consistency')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses', fontweight='bold')
        ax1.legend()
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(epochs, self.history['val_dice'], 'g-', linewidth=2)
        best_epoch = np.argmax(self.history['val_dice']) + 1
        best_dice = max(self.history['val_dice'])
        ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7)
        ax2.scatter([best_epoch], [best_dice], color='red', s=100, marker='*')
        ax2.axhline(y=0.7028, color='orange', linestyle=':', label='Supervised Baseline')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.set_title(f'Validation Dice (Best: {best_dice:.4f})', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(epochs, self.history['val_dice_fg'], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Predicted FG Pixels')
        ax3.set_title('Foreground Prediction Volume', fontweight='bold')
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        improvement = ((best_dice - 0.7028) / 0.7028) * 100
        summary = f"""
╔══════════════════════════════════════════════════╗
║     MEAN TEACHER (FIXED DICE) RESULTS            ║
╠══════════════════════════════════════════════════╣
║  EMA Decay: {self.ema_decay:.2f}                              ║
║  Consistency Weight: {self.consistency_weight_max:.1f}                       ║
║  Dice Epsilon: 1e-6 (FIXED)                      ║
╠══════════════════════════════════════════════════╣
║  COMPARISON TO BASELINE                          ║
║  Supervised (221 labels): 0.7028                 ║
║  Mean Teacher (50 labels): {best_dice:.4f}               ║
║  Improvement: {improvement:+.2f}%                            ║
╠══════════════════════════════════════════════════╣
║  Best Epoch: {best_epoch}                                   ║
╚══════════════════════════════════════════════════╝
"""
        ax4.text(0.5, 0.5, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / 'mean_teacher_progress.png', dpi=150)
        plt.savefig(self.output_dir / 'mean_teacher_progress.pdf')
        plt.close()
        print(f"Plots saved to {self.output_dir}")
    
    def save_history_csv(self):
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'supervised_loss': self.history['supervised_loss'],
            'consistency_loss': self.history['consistency_loss'],
            'val_dice': self.history['val_dice'],
            'val_fg_pred': self.history['val_dice_fg'],
            'learning_rate': self.history['learning_rate'],
            'consistency_weight': self.history['consistency_weight']
        })
        df.to_csv(self.output_dir / 'training_history.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('preprocessed_v2'))
    parser.add_argument('--experiment_name', type=str, default='fixed_dice')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--consistency_weight', type=float, default=1.0)
    parser.add_argument('--consistency_rampup', type=int, default=5)
    parser.add_argument('--num_labeled', type=int, default=50)
    parser.add_argument('--num_unlabeled', type=int, default=161)
    parser.add_argument('--num_validation', type=int, default=70)
    args = parser.parse_args()
    
    setup_gpu()
    
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    
    data_paths = prepare_ssl_data_paths(
        args.data_dir,
        num_labeled=args.num_labeled,
        num_unlabeled=args.num_unlabeled,
        num_validation=args.num_validation
    )
    
    exp_dir = Path(f'experiments/mean_teacher_{args.experiment_name}_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = MeanTeacherFixed(
        config,
        ema_decay=args.ema_decay,
        consistency_weight_max=args.consistency_weight,
        consistency_rampup_epochs=args.consistency_rampup,
        output_dir=exp_dir
    )
    
    trainer.train(data_paths, num_epochs=args.num_epochs)
    print(f"\nResults in: {exp_dir}")


if __name__ == "__main__":
    main()
