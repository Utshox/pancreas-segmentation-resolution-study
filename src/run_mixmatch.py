"""
MixMatch Semi-Supervised Learning Runner
Uses pseudo-labeling with MixUp augmentation
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


class MixMatchTrainer:
    """MixMatch SSL with pseudo-labeling and MixUp"""
    
    def __init__(self, config, temperature=0.5, alpha=0.75, lambda_u=1.0, 
                 rampup_epochs=5, output_dir=None):
        self.config = config
        self.temperature = temperature
        self.alpha = alpha
        self.lambda_u_max = lambda_u
        self.rampup_epochs = rampup_epochs
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        print(f"\n=== MixMatch Trainer ===")
        print(f"  Temperature: {temperature}")
        print(f"  MixUp Alpha: {alpha}")
        print(f"  Unlabeled Weight Max: {lambda_u}")
        
        self.model = PancreasSeg(config)
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.model(dummy_input)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.supervised_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.unsupervised_loss = tf.keras.losses.MeanSquaredError()
        
        self.data_pipeline = DataPipeline(config)
        
        self.history = {
            'train_loss': [], 'supervised_loss': [], 'unsupervised_loss': [],
            'val_dice': [], 'learning_rate': [], 'lambda_u': []
        }
    
    def _sharpen(self, p, T):
        """Sharpen probability distribution"""
        p_pow = tf.pow(p, 1.0/T)
        return p_pow / (p_pow + tf.pow(1-p, 1.0/T) + 1e-7)
    
    def _mixup(self, x1, x2, y1, y2, alpha):
        """MixUp augmentation"""
        batch_size = tf.shape(x1)[0]
        lam = tf.random.uniform([batch_size, 1, 1, 1], minval=0, maxval=1)
        lam = tf.maximum(lam, 1 - lam)  # Ensure lam >= 0.5
        
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        
        return x_mixed, y_mixed
    
    @tf.function
    def _augment(self, images):
        """Apply strong augmentation"""
        noise = tf.random.normal(tf.shape(images), mean=0.0, stddev=0.1)
        images = images + noise
        images = images * tf.random.uniform([], 0.8, 1.2)
        return tf.clip_by_value(images, -1.0, 1.0)
    
    def _get_lambda_u(self, epoch):
        if epoch >= self.rampup_epochs:
            return self.lambda_u_max
        return self.lambda_u_max * (epoch / self.rampup_epochs)
    
    @tf.function
    def train_step(self, labeled_images, labeled_labels, unlabeled_images, lambda_u):
        # Generate pseudo-labels for unlabeled data
        unlabeled_logits = self.model(unlabeled_images, training=False)
        pseudo_probs = tf.nn.sigmoid(unlabeled_logits)
        pseudo_labels = self._sharpen(pseudo_probs, self.temperature)
        
        # Augment both labeled and unlabeled
        labeled_aug = self._augment(labeled_images)
        unlabeled_aug = self._augment(unlabeled_images)
        
        # Combine all data
        all_images = tf.concat([labeled_aug, unlabeled_aug], axis=0)
        all_labels = tf.concat([labeled_labels, pseudo_labels], axis=0)
        
        # Shuffle
        batch_size = tf.shape(labeled_images)[0]
        total_size = tf.shape(all_images)[0]
        indices = tf.random.shuffle(tf.range(total_size))
        all_images_shuffled = tf.gather(all_images, indices)
        all_labels_shuffled = tf.gather(all_labels, indices)
        
        # MixUp
        mixed_images, mixed_labels = self._mixup(
            all_images, all_images_shuffled, 
            all_labels, all_labels_shuffled, 
            self.alpha
        )
        
        with tf.GradientTape() as tape:
            logits = self.model(mixed_images, training=True)
            probs = tf.nn.sigmoid(logits)
            
            # Split back to labeled and unlabeled
            labeled_logits = logits[:batch_size]
            unlabeled_probs = probs[batch_size:]
            
            labeled_mixed_labels = mixed_labels[:batch_size]
            unlabeled_mixed_labels = mixed_labels[batch_size:]
            
            # Supervised loss (BCE on labeled)
            sup_loss = self.supervised_loss(labeled_mixed_labels, labeled_logits)
            
            # Unsupervised loss (MSE on unlabeled pseudo-labels)
            unsup_loss = self.unsupervised_loss(unlabeled_mixed_labels, unlabeled_probs)
            
            total_loss = sup_loss + lambda_u * unsup_loss
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, sup_loss, unsup_loss
    
    def compute_dice(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)
    
    def validate(self, val_dataset):
        dice_scores = []
        for batch in val_dataset:
            images, labels = batch
            predictions = self.model(images, training=False)
            if len(labels.shape) > 3:
                labels = labels[..., -1]
            if len(predictions.shape) > 3:
                predictions = predictions[..., -1]
            dice = self.compute_dice(labels, predictions)
            dice_scores.append(float(dice))
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def train(self, data_paths, num_epochs=30):
        print(f"\nStarting MixMatch training for {num_epochs} epochs...")
        
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
            epoch_unsup_losses = []
            
            lambda_u = self._get_lambda_u(epoch)
            
            if unlabeled_ds:
                combined_ds = tf.data.Dataset.zip((labeled_ds, unlabeled_ds.repeat()))
            else:
                combined_ds = labeled_ds
            
            current_lr = float(self.optimizer.learning_rate)
            print(f"\nEpoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}, λu: {lambda_u:.2f}]")
            progress = tqdm(combined_ds, desc="Training")
            
            for batch in progress:
                if unlabeled_ds:
                    (labeled_imgs, labeled_lbls), unlabeled_imgs = batch
                    if isinstance(unlabeled_imgs, tuple):
                        unlabeled_imgs = unlabeled_imgs[0]
                else:
                    labeled_imgs, labeled_lbls = batch
                    unlabeled_imgs = labeled_imgs
                
                total_loss, sup_loss, unsup_loss = self.train_step(
                    labeled_imgs, labeled_lbls, unlabeled_imgs,
                    tf.constant(lambda_u, dtype=tf.float32)
                )
                
                epoch_losses.append(float(total_loss))
                epoch_sup_losses.append(float(sup_loss))
                epoch_unsup_losses.append(float(unsup_loss))
                
                progress.set_postfix({
                    'loss': f'{float(total_loss):.4f}',
                    'sup': f'{float(sup_loss):.4f}',
                    'unsup': f'{float(unsup_loss):.4f}'
                })
            
            val_dice = self.validate(val_ds)
            epoch_time = time.time() - start_time
            
            avg_loss = np.mean(epoch_losses)
            avg_sup = np.mean(epoch_sup_losses)
            avg_unsup = np.mean(epoch_unsup_losses)
            
            self.history['train_loss'].append(avg_loss)
            self.history['supervised_loss'].append(avg_sup)
            self.history['unsupervised_loss'].append(avg_unsup)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(current_lr)
            self.history['lambda_u'].append(lambda_u)
            
            print(f"Time: {epoch_time:.1f}s | Loss: {avg_loss:.4f} | Sup: {avg_sup:.4f} | Unsup: {avg_unsup:.4f} | Val Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_mixmatch')
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
        checkpoint_dir = Path('mixmatch_results/checkpoints') / time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(checkpoint_dir / f'{name}.weights.h5'))
        print(f"Saved to {checkpoint_dir}")
    
    def plot_progress(self):
        if len(self.history['train_loss']) < 2:
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('MixMatch Training Progress', fontweight='bold', fontsize=16, y=0.98)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Total')
        ax1.plot(epochs, self.history['supervised_loss'], 'g--', linewidth=1.5, label='Supervised')
        ax1.plot(epochs, self.history['unsupervised_loss'], 'r--', linewidth=1.5, label='Unsupervised')
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
        ax3.fill_between(epochs, self.history['lambda_u'], alpha=0.3, color='purple')
        ax3.plot(epochs, self.history['lambda_u'], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('λu')
        ax3.set_title('Unsupervised Loss Weight', fontweight='bold')
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        improvement = ((best_dice - 0.7028) / 0.7028) * 100
        summary = f"""
╔══════════════════════════════════════════════════╗
║          MIXMATCH RESULTS                        ║
╠══════════════════════════════════════════════════╣
║  Temperature: {self.temperature:.2f}                              ║
║  MixUp Alpha: {self.alpha:.2f}                              ║
║  Unlabeled Weight: {self.lambda_u_max:.1f}                         ║
╠══════════════════════════════════════════════════╣
║  COMPARISON TO BASELINE                          ║
║  Supervised (221 labels): 0.7028                 ║
║  MixMatch (50 labels): {best_dice:.4f}                 ║
║  Improvement: {improvement:+.2f}%                            ║
╠══════════════════════════════════════════════════╣
║  Best Epoch: {best_epoch}                                   ║
╚══════════════════════════════════════════════════╝
"""
        ax4.text(0.5, 0.5, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / 'mixmatch_progress.png', dpi=150)
        plt.savefig(self.output_dir / 'mixmatch_progress.pdf')
        plt.close()
        print(f"Plots saved to {self.output_dir}")
    
    def save_history_csv(self):
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'supervised_loss': self.history['supervised_loss'],
            'unsupervised_loss': self.history['unsupervised_loss'],
            'val_dice': self.history['val_dice'],
            'learning_rate': self.history['learning_rate'],
            'lambda_u': self.history['lambda_u']
        })
        df.to_csv(self.output_dir / 'training_history.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('preprocessed_v2'))
    parser.add_argument('--experiment_name', type=str, default='mixmatch')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--rampup', type=int, default=5)
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
    
    exp_dir = Path(f'experiments/mixmatch_{args.experiment_name}_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = MixMatchTrainer(
        config,
        temperature=args.temperature,
        alpha=args.alpha,
        lambda_u=args.lambda_u,
        rampup_epochs=args.rampup,
        output_dir=exp_dir
    )
    
    trainer.train(data_paths, num_epochs=args.num_epochs)
    print(f"\nResults in: {exp_dir}")


if __name__ == "__main__":
    main()
