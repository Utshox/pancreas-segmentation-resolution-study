"""
Diagnostic Script: Check if validation Dice is truly constant or a bug
This will:
1. Create model and train for a few steps
2. Print actual predictions stats at different points
3. Verify Dice calculation is working correctly
"""
import sys
import os
import numpy as np
from pathlib import Path

sys.path.append('/scratch/lustre/home/mdah0000/smm/v14')

import tensorflow as tf
from config import StableSSLConfig
from train_ssl_tf2n import setup_gpu
from data_loader_tf2 import DataPipeline
from models_tf2 import PancreasSeg


def compute_dice_debug(y_true, y_pred_logits):
    """Compute Dice with debug output"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.nn.sigmoid(y_pred_logits)
    y_pred_binary = tf.cast(tf.squeeze(y_pred) > 0.5, tf.float32)
    
    # Flatten
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred_binary, [tf.shape(y_pred_binary)[0], -1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    sum_true = tf.reduce_sum(y_true_flat, axis=1)
    sum_pred = tf.reduce_sum(y_pred_flat, axis=1)
    
    dice = (2.0 * intersection + 1.0) / (sum_true + sum_pred + 1.0)
    
    return dice, {
        'intersection': intersection.numpy(),
        'sum_true': sum_true.numpy(),
        'sum_pred': sum_pred.numpy(),
        'pred_probs_mean': tf.reduce_mean(y_pred).numpy(),
        'pred_probs_std': tf.math.reduce_std(y_pred).numpy(),
        'pred_min': tf.reduce_min(y_pred).numpy(),
        'pred_max': tf.reduce_max(y_pred).numpy()
    }


def prepare_data_paths(data_dir, num_validation=10):
    """Get just validation data for quick test"""
    def get_case_number(path):
        name = path.name.replace('.nii', '')
        return int(name.split('pancreas_')[-1][:3])
    
    all_folders = sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x))
    
    # Use last num_validation folders for validation
    val_folders = all_folders[-num_validation:]
    
    images, labels = [], []
    for folder in val_folders:
        img_path = folder / 'image.npy'
        mask_path = folder / 'mask.npy'
        if img_path.exists() and mask_path.exists():
            images.append(str(img_path))
            labels.append(str(mask_path))
    
    return images, labels


def main():
    print("="*60)
    print("VALIDATION DICE DIAGNOSTIC TEST")
    print("="*60)
    
    setup_gpu()
    
    config = StableSSLConfig()
    config.batch_size = 4
    
    data_dir = Path('preprocessed_v2')
    val_images, val_labels = prepare_data_paths(data_dir, num_validation=10)
    print(f"\nUsing {len(val_images)} validation patients")
    
    # Create model
    print("\nCreating model...")
    model = PancreasSeg(config)
    dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
    _ = model(dummy_input)
    
    # Create validation dataset
    data_pipeline = DataPipeline(config)
    val_ds = data_pipeline.build_validation_dataset(val_images, val_labels, batch_size=4)
    
    # Test 1: Check predictions with untrained model
    print("\n" + "="*60)
    print("TEST 1: Untrained Model Predictions")
    print("="*60)
    
    for i, (images, labels) in enumerate(val_ds.take(3)):
        preds = model(images, training=False)
        dice, stats = compute_dice_debug(labels, preds)
        print(f"\nBatch {i+1}:")
        print(f"  Dice values: {dice.numpy()}")
        print(f"  Mean Dice: {tf.reduce_mean(dice).numpy():.6f}")
        print(f"  Pred probs - mean: {stats['pred_probs_mean']:.4f}, std: {stats['pred_probs_std']:.4f}")
        print(f"  Pred range: [{stats['pred_min']:.4f}, {stats['pred_max']:.4f}]")
    
    # Test 2: Train for a few steps and check again
    print("\n" + "="*60)
    print("TEST 2: After 50 Training Steps")
    print("="*60)
    
    # Quick training
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Get some training data
    train_images = val_images[:5]  # Use first 5 val images as "training" for this test
    train_labels = val_labels[:5]
    train_ds = data_pipeline.build_labeled_dataset(train_images, train_labels, batch_size=4, is_training=True)
    
    print("\nTraining for 50 steps...")
    for step, (images, labels) in enumerate(train_ds.repeat().take(50)):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss = loss_fn(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 10 == 0:
            print(f"  Step {step}, Loss: {loss.numpy():.4f}")
    
    print("\nValidation after training:")
    for i, (images, labels) in enumerate(val_ds.take(3)):
        preds = model(images, training=False)
        dice, stats = compute_dice_debug(labels, preds)
        print(f"\nBatch {i+1}:")
        print(f"  Dice values: {dice.numpy()}")
        print(f"  Mean Dice: {tf.reduce_mean(dice).numpy():.6f}")
        print(f"  Pred probs - mean: {stats['pred_probs_mean']:.4f}, std: {stats['pred_probs_std']:.4f}")
    
    # Test 3: Train more and check if Dice changes
    print("\n" + "="*60)
    print("TEST 3: After 100 More Training Steps")
    print("="*60)
    
    print("\nTraining for 100 more steps...")
    for step, (images, labels) in enumerate(train_ds.repeat().take(100)):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss = loss_fn(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 20 == 0:
            print(f"  Step {step}, Loss: {loss.numpy():.4f}")
    
    print("\nValidation after more training:")
    all_dices = []
    for i, (images, labels) in enumerate(val_ds.take(3)):
        preds = model(images, training=False)
        dice, stats = compute_dice_debug(labels, preds)
        all_dices.extend(dice.numpy().tolist())
        print(f"\nBatch {i+1}:")
        print(f"  Dice values: {dice.numpy()}")
        print(f"  Mean Dice: {tf.reduce_mean(dice).numpy():.6f}")
        print(f"  Pred probs - mean: {stats['pred_probs_mean']:.4f}, std: {stats['pred_probs_std']:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"All Dice values: {all_dices}")
    print(f"Are all identical? {len(set([round(d, 6) for d in all_dices])) == 1}")
    print(f"Unique values: {len(set([round(d, 6) for d in all_dices]))}")
    
    print("\n✓ Diagnostic complete!")


if __name__ == "__main__":
    main()
