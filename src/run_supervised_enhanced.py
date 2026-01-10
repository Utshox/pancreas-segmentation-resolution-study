import sys
import os
import numpy as np

sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

import tensorflow as tf
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import StableSSLConfig, ExperimentConfig
from train_ssl_tf2n import SupervisedTrainer, StableDiceLoss
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg
from main import prepare_data_paths, setup_gpu

class EnhancedSupervisedTrainer(SupervisedTrainer):
    """Enhanced supervised trainer with PROPERLY calculated LR schedule"""
    
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.data_pipeline = DataPipeline(config)
        self._setup_model()
        # Note: LR schedule will be set up AFTER we count batches in first epoch
        self.optimizer = None
        self.lr_schedule = None
        self.loss_fn = StableDiceLoss(smooth=1.0)
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
    def _setup_model(self):
        print("Setting up enhanced U-Net model...")
        self.model = PancreasSeg(self.config)
        dummy_input = tf.zeros((1, self.config.img_size_x, self.config.img_size_y, self.config.num_channels))
        _ = self.model(dummy_input)
    
    def _setup_optimizer_with_steps(self, steps_per_epoch):
        """Setup optimizer AFTER counting actual batches"""
        total_steps = self.config.num_epochs * steps_per_epoch
        
        print(f"\n=== LR SCHEDULE (PROPERLY CALCULATED) ===")
        print(f"Counted batches per epoch: {steps_per_epoch}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Total decay_steps: {total_steps}")
        print(f"LR will decay from 1e-3 to 1e-5 over {self.config.num_epochs} epochs")
        print(f"==========================================\n")
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=total_steps,
            alpha=0.01
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
    
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss = self.loss_fn(labels, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, logits
    
    def compute_dice(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.nn.sigmoid(y_pred) > 0.5, tf.float32)
        
        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return tf.reduce_mean(dice)
    
    def validate(self, val_dataset):
        dice_scores = []
        for batch in val_dataset:
            images, labels = batch
            predictions = self.model(images, training=False)
            dice_score = self.compute_dice(labels, predictions)
            dice_scores.append(float(dice_score))
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def train(self, data_paths):
        print("\nStarting training with PROPERLY calculated LR schedule...")
        
        # Create datasets WITH CACHING
        train_ds = self.data_pipeline.build_labeled_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            is_training=True
        ).cache()
        
        val_ds = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size
        ).cache()
        
        # Count actual batches in first epoch
        print("Counting actual batches per epoch...")
        batch_count = 0
        for _ in train_ds:
            batch_count += 1
        print(f"Found {batch_count} batches per epoch")
        
        # NOW setup optimizer with correct decay_steps
        self._setup_optimizer_with_steps(batch_count)
        
        best_dice = 0
        patience = self.config.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            current_lr = float(self.lr_schedule(self.optimizer.iterations))
            
            epoch_losses = []
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} [LR: {current_lr:.6f}]")
            progress_bar = tqdm(train_ds, desc="Training")
            
            for batch in progress_bar:
                images, labels = batch
                loss, _ = self.train_step(images, labels)
                epoch_losses.append(float(loss))
                progress_bar.set_postfix({"loss": f"{float(loss):.4f}"})
            
            val_dice = self.validate(val_ds)
                
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(float(sum(epoch_losses) / len(epoch_losses)))
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Time: {epoch_time:.2f}s | Loss: {self.history['train_loss'][-1]:.4f} | Val Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_supervised')
                print(f"✓ New best model saved! Dice: {best_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
            
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        self.plot_progress()
        
        return self.history
    
    def save_checkpoint(self, name):
        checkpoint_dir = Path('supervised_results/checkpoints') / time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / f'{name}.weights.h5'
        self.model.save_weights(str(model_path))
        np.save(checkpoint_dir / f'{name}_history.npy', self.history)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    def plot_progress(self):
        if len(self.history['train_loss']) < 2:
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
            'legend.fontsize': 11, 'figure.titlesize': 18, 'lines.linewidth': 2,
        })
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Supervised Baseline (LR PROPERLY FIXED)', fontweight='bold', y=0.98)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', alpha=0.3, label='Raw Loss')
        if len(self.history['train_loss']) > 5:
            smoothed = self._smooth_curve(self.history['train_loss'], factor=0.9)
            ax1.plot(epochs, smoothed, 'b-', linewidth=2.5, label='Smoothed Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dice Loss')
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_ylim(bottom=0)
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(epochs, self.history['val_dice'], 'g-', linewidth=2, label='Validation Dice')
        best_epoch = np.argmax(self.history['val_dice']) + 1
        best_dice = max(self.history['val_dice'])
        ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
        ax2.scatter([best_epoch], [best_dice], color='red', s=100, zorder=5, marker='*')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.set_title('Validation Dice Score', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 1)
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(epochs, self.history['learning_rate'], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule (Cosine Decay)', fontweight='bold')
        ax3.set_yscale('log')
        ax3.fill_between(epochs, self.history['learning_rate'], alpha=0.3, color='purple')
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        final_loss = self.history['train_loss'][-1]
        final_dice = self.history['val_dice'][-1]
        total_epochs = len(self.history['train_loss'])
        
        summary_text = f"""
╔══════════════════════════════════════════════╗
║     TRAINING SUMMARY                         ║
╠══════════════════════════════════════════════╣
║  Experiment: Supervised Baseline             ║
║  Total Epochs: {total_epochs:>3}                            ║
╠══════════════════════════════════════════════╣
║  FINAL METRICS                               ║
║  Final Training Loss:    {final_loss:>8.4f}             ║
║  Final Validation Dice:  {final_dice:>8.4f}             ║
╠══════════════════════════════════════════════╣
║  BEST MODEL                                  ║
║  Best Validation Dice:   {best_dice:>8.4f}             ║
║  Best Epoch:             {best_epoch:>8}             ║
╚══════════════════════════════════════════════╝
"""
        ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path = self.output_dir / 'training_progress.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
        
        pdf_path = self.output_dir / 'training_progress.pdf'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"PDF saved to {pdf_path}")
        
        plt.close()
        self._save_history_csv()
    
    def _smooth_curve(self, values, factor=0.9):
        smoothed = []
        last = values[0]
        for v in values:
            smoothed_val = last * factor + v * (1 - factor)
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def _save_history_csv(self):
        import pandas as pd
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_dice': self.history['val_dice'],
            'learning_rate': self.history['learning_rate']
        })
        csv_path = self.output_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        print(f"Training history saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=False,
                      default='/scratch/lustre/home/mdah0000/images/cropped')
    parser.add_argument('--experiment_name', type=str, default='supervised_proper_lr')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()

    setup_gpu()
    
    config = StableSSLConfig()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type='supervised',
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )
    
    print("Preparing data paths...")
    total_files = len(list(args.data_dir.glob('pancreas_*/image.npy')))
    
    num_val = 60
    num_lab = max(1, total_files - num_val)
    
    print(f"Data split: {num_lab} training, {num_val} validation")
    
    data_paths = prepare_data_paths(args.data_dir, num_labeled=num_lab, num_validation=num_val)
    
    print("Creating enhanced supervised trainer...")
    exp_dir = experiment_config.get_experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)

    trainer = EnhancedSupervisedTrainer(config, output_dir=exp_dir)
    
    print("\nExperiment Configuration:")
    print(f"Training Type: supervised (PROPER LR)")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Image size: {config.img_size_x}x{config.img_size_y}")
    print(f"Dataset CACHING: ENABLED")
    
    trainer.train(data_paths)

if __name__ == "__main__":
    main()
