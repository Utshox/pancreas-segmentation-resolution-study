"""
FixMatch SSL Algorithm (Adapted for Binary Medical Segmentation)
Backbone: U-Net (Baseline)
Loss: Supervised (GEMINI) + Unsupervised (Consistency w/ Thresholding)
Data: V3 (Strict HU Windowing)
"""
import sys
import os
import numpy as np
import time
from pathlib import Path
import argparse
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

from config import StableSSLConfig
from train_ssl_tf2n import setup_gpu
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg 

def get_all_data_paths(data_dir):
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])
    
    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    train_folders = all_folders[:221] 
    val_folders = all_folders[221:]
    
    # 20% Labeled Split (~44 patients)
    n_labeled = 44 
    labeled_folders = train_folders[:n_labeled]
    unlabeled_folders = train_folders[n_labeled:]
    
    print(f"Data Split: {len(labeled_folders)} Labeled, {len(unlabeled_folders)} Unlabeled, {len(val_folders)} Validation")
    
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
        'unlabeled': dict(zip(['images', 'labels'], get_paths(unlabeled_folders))),
        'validation': dict(zip(['images', 'labels'], get_paths(val_folders)))
    }

def weak_augment(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    return images

def strong_augment(images):
    # REDUCED INTENSITY (V3 Compatible)
    # Brightness 0.1, Contrast [0.8, 1.2]
    images = tf.image.random_brightness(images, max_delta=0.1)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=0.05, dtype=tf.float32)
    images = images + noise
    images = tf.clip_by_value(images, 0.0, 1.0)
    return images

class FixMatchTrainer:
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        self.model = PancreasSeg(config)
        _ = self.model(tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels)))
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, 50000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.data_pipeline = DataPipeline(config)
        self.history = {'train_loss': [], 'val_dice': [], 'ssl_loss': []}
        
    def dice_coef(self, y_true, y_pred):
        smooth = 1e-6
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def binary_focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        y_pred_f = tf.clip_by_value(y_pred_f, 1e-7, 1.0 - 1e-7)
        pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
        pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))
        return -tf.reduce_mean(alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1)) \
               -tf.reduce_mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0))

    def gemini_loss(self, y_true, y_pred):
        dice_loss = 1.0 - self.dice_coef(y_true, y_pred)
        focal_l = self.binary_focal_loss(y_true, y_pred)
        return 0.5 * dice_loss + 0.5 * focal_l
    
    @tf.function
    def train_step(self, x_labeled, y_labeled, x_unlabeled, warmup=False):
        with tf.GradientTape() as tape:
            # 1. Supervised 
            logits_l = self.model(x_labeled, training=True)
            pred_l = tf.nn.sigmoid(logits_l) 
            loss_sup = self.gemini_loss(y_labeled, pred_l)
            
            loss_u = 0.0
            
            if not warmup:
                # 2. FixMatch
                x_u_w = weak_augment(x_unlabeled)
                logits_u_w = self.model(x_u_w, training=False) 
                pred_u_w = tf.nn.sigmoid(logits_u_w) 
                
                x_u_s = strong_augment(x_unlabeled)
                logits_u_s = self.model(x_u_s, training=True)
                pred_u_s = tf.nn.sigmoid(logits_u_s)
                
                # LOWER THRESHOLD 0.80
                high_conf_1 = tf.cast(tf.math.greater(pred_u_w, 0.80), tf.float32)
                high_conf_0 = tf.cast(tf.math.less(pred_u_w, 0.20), tf.float32)
                mask = high_conf_1 + high_conf_0
                
                pseudo_label = high_conf_1 
                loss_u = self.bce(pseudo_label, pred_u_s, sample_weight=mask)
            
            total_loss = loss_sup + loss_u
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_sup, loss_u
    
    def validate(self, val_ds):
        dices = []
        for x, y in val_ds:
            logits = self.model(x, training=False)
            pred = tf.nn.sigmoid(logits) 
            d = self.dice_coef(y, pred)
            dices.append(float(d))
        return np.mean(dices)

    def train(self, data_paths, num_epochs, warmup_epochs=0):
        print(f"Starting FixMatch for {num_epochs} epochs (Warmup: {warmup_epochs})...")
        
        ds_l = self.data_pipeline.build_labeled_dataset(
            data_paths['labeled']['images'], data_paths['labeled']['labels'],
            batch_size=self.config.batch_size
        )
        ds_u = self.data_pipeline.build_unlabeled_dataset_for_mixmatch(
            data_paths['unlabeled']['images'], 
            batch_size=self.config.batch_size
        )
        ds_l_rep = ds_l.repeat()
        train_ds = tf.data.Dataset.zip((ds_l_rep, ds_u))
        val_ds = self.data_pipeline.build_validation_dataset(
            data_paths['validation']['images'], data_paths['validation']['labels'],
            batch_size=self.config.batch_size
        )
        
        best_dice = 0.0
        patience_counter = 0
        patience_limit = 20 # Increased for V3
        
        for epoch in range(num_epochs):
            losses_sup = []
            losses_ssl = []
            
            is_warmup = epoch < warmup_epochs
            
            for (x_l, y_l), (x_u,) in tqdm(train_ds, desc=f"Epoch {epoch+1} {'(Warmup)' if is_warmup else ''}"):
                ls, lu = self.train_step(x_l, y_l, x_u, warmup=is_warmup)
                losses_sup.append(float(ls))
                losses_ssl.append(float(lu))
                
            val_dice = self.validate(val_ds)
            print(f"Epoch {epoch+1}: Val Dice: {val_dice:.4f} | Sup Loss: {np.mean(losses_sup):.4f} | SSL Loss: {np.mean(losses_ssl):.4f}")
            
            self.history['train_loss'].append(np.mean(losses_sup))
            self.history['ssl_loss'].append(np.mean(losses_ssl))
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
        plt.plot(self.history['train_loss'], label='Sup Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'fixmatch_progress.png')
        plt.close()

    def save_history(self):
        pd.DataFrame(self.history).to_csv(self.output_dir / 'fixmatch_history.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    args = parser.parse_args()
    
    setup_gpu()
    config = StableSSLConfig()
    config.batch_size = args.batch_size
    
    exp_dir = Path(f'experiments/fixmatch_v3_{time.strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # USE V3 DATA
    paths = get_all_data_paths(Path('preprocessed_v3'))
    
    trainer = FixMatchTrainer(config, output_dir=exp_dir)
    trainer.train(paths, args.num_epochs, warmup_epochs=args.warmup_epochs)

if __name__ == "__main__":
    main()
