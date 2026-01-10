
import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np
from pathlib import Path
import time
import logging
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from models_tf2 import *
# from visualization import ExperimentVisualizer # Missing file

from data_loader_tf2 import DataPipeline, PancreasDataLoader
# from visualization import ExperimentVisualizer


AUTOTUNE = tf.data.experimental.AUTOTUNE
#mean_teacher
#GPU
def setup_gpu():
    """Setup GPU for training"""
    print("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Use all available GPUs and allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Log GPU information
            print(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  {gpu.device_type}: {gpu.name}")
                
            # Set mixed precision policy
            tf.keras.mixed_precision.set_global_policy('float32')  # Changed to float32 for stability
            print("Using float32 precision")
            
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPU found. Using CPU.")
        return False

def prepare_data_paths(data_dir, num_labeled=20, num_validation=63):
    """Prepare data paths without verbose logging"""
    def get_case_number(path):
        return int(str(path).split('pancreas_')[-1][:3])

    # Gather paths silently
    all_image_paths = []
    all_label_paths = []
    
    for folder in sorted(data_dir.glob('pancreas_*'), key=lambda x: get_case_number(x)):
        folder_no_nii = str(folder).replace('.nii', '')
        img_path = Path(folder_no_nii) / 'img_cropped.npy'
        mask_path = Path(folder_no_nii) / 'mask_cropped.npy'
        
        if img_path.exists() and mask_path.exists():
            all_image_paths.append(str(img_path))
            all_label_paths.append(str(mask_path))

    total_samples = len(all_image_paths)
    print(f"Found {total_samples} valid samples")
    
    if total_samples < (num_labeled + num_validation):
        raise ValueError(f"Not enough valid samples. Found {total_samples}, need at least {num_labeled + num_validation}")

    # Create splits
    train_images = all_image_paths[:num_labeled]
    train_labels = all_label_paths[:num_labeled]
    val_images = all_image_paths[-num_validation:]
    val_labels = all_label_paths[-num_validation:]
    unlabeled_images = all_image_paths[num_labeled:-num_validation]

    print(f"Data split - Labeled: {len(train_images)}, Unlabeled: {len(unlabeled_images)}, Validation: {len(val_images)}")

    return {
        'labeled': {
            'images': train_images,
            'labels': train_labels,
        },
        'unlabeled': {
            'images': unlabeled_images
        },
        'validation': {
            'images': val_images,
            'labels': val_labels,
        }
    }


def masked_binary_crossentropy(y_true, y_pred):
    """Binary crossentropy that ignores None values"""
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Apply sigmoid if needed (for logits)
    y_pred = tf.nn.sigmoid(y_pred)
    
    # Calculate binary crossentropy
    bce = -(y_true * tf.math.log(y_pred + 1e-7) + 
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
    
    # Return mean of non-zero elements
    return tf.reduce_mean(bce)





#### 14jun 

class StableDiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]

        y_pred = tf.nn.sigmoid(y_pred)
        
        reduction_axes = [1, 2]
        intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_axes)
        sum_true = tf.reduce_sum(y_true, axis=reduction_axes)
        sum_pred = tf.reduce_sum(y_pred, axis=reduction_axes)

        dice = (2. * intersection + self.smooth) / (sum_true + sum_pred + self.smooth)
        dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
        dice = tf.where(tf.math.is_inf(dice), tf.ones_like(dice), dice)
        
        return 1.0 - tf.reduce_mean(dice)

#############################################################
class SupervisedTrainer:
    def __init__(self, config):
        """Initialize supervised trainer"""
        print("Initializing Supervised Trainer...")
        self.config = config

        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # Initialize model
        self.model = PancreasSeg(config)
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.model(dummy_input)
        
        # Setup training parameters
        self._setup_training_params()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        print("Supervised trainer initialization complete!")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('supervised_results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def _setup_training_params(self):
        """Setup training parameters"""
        initial_lr = 5e-4  # Higher learning rate for supervised
        
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=200,
            t_mul=1.5,
            m_mul=0.95,
            alpha=0.1
        )
        
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.99
        )
        
        self.loss_fn = StableDiceLoss(smooth=5.0)

    @tf.function
    def train_step(self, images, labels):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(images, training=True)
            
            # Calculate loss
            loss = self.loss_fn(labels, logits)
        
        # Calculate gradients and apply updates
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, logits

    def compute_dice(self, y_true, y_pred):
        """Compute Dice score"""
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
        """Run validation"""
        dice_scores = []
        
        for batch in val_dataset:
            images, labels = batch
            predictions = self.model(images, training=False)
            dice_score = self.compute_dice(labels, predictions)
            dice_scores.append(float(dice_score))
        
        return np.mean(dice_scores)

    def train(self, data_paths):
        """Main training loop"""
        print("\nStarting supervised training...")
        
        # Create datasets
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        best_dice = 0
        patience = self.config.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training
            epoch_losses = []
            for batch in train_ds:
                images, labels = batch
                loss, _ = self.train_step(images, labels)
                epoch_losses.append(float(loss))
            
            # Validation every 2 epochs
            if epoch % 2 == 0:
                val_dice = self.validate(val_ds)
                
                # Update history
                self.history['val_dice'].append(val_dice)
                
                # Save best model
                if val_dice > best_dice:
                    best_dice = val_dice
                    self.save_checkpoint('best_supervised_model')
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!")
                    break
            
            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Time: {time.time() - start_time:.2f}s | "
                  f"Loss: {np.mean(epoch_losses):.4f}"
                  + (f" | Val Dice: {val_dice:.4f}" if epoch % 2 == 0 else ""))
            
            # Plot progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_progress()
        
        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
        return self.history

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.checkpoint_dir / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / f'{name}.weights.h5'
        self.model.save_weights(str(model_path))
        # Save history
        np.save(checkpoint_dir / f'{name}_history.npy', self.history)
        print(f"Saved checkpoint to {checkpoint_dir}")

    def plot_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_dice'])
        plt.title('Validation Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'training_progress_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()


#############################################################




class StableSSLTrainer:
    def __init__(self, config, labeled_data_size=2):
        print("Initializing StableSSLTrainer...")
        self.config = config
        self.labeled_data_size = labeled_data_size
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(config)
        
        # Initialize models
        self.student_model = PancreasSeg(config)
        self.teacher_model = PancreasSeg(config)
        
        # Initialize with dummy input
        dummy_input = tf.zeros((1, config.img_size_x, config.img_size_y, config.num_channels))
        _ = self.student_model(dummy_input)
        _ = self.teacher_model(dummy_input)
        
        # Copy initial weights
        self.teacher_model.set_weights(self.student_model.get_weights())
        
        # Setup directories
        self.setup_directories()
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_dice': [],
            'teacher_dice': [],
            'learning_rate': [],
            'supervised_loss': [],
            'consistency_loss': []
        }
        
        print("Initialization complete!")

    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path('ssl_results')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def _setup_training_params(self):
        """Setup training parameters with more dynamic learning"""
        # Higher initial learning rate with better decay parameters
        initial_lr = 5e-4
        
        # CosineDecayRestarts schedule with minimum learning rate
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=500,  # Adjust based on your dataset size
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1,  # Setting alpha to 0.1 ensures learning rate never goes below 10% of initial
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        # Loss function with larger smooth factor
        self.supervised_loss_fn = StableDiceLoss(smooth=10.0)
        
        # Consistency weight parameters
        self.initial_consistency_weight = 0.0
        self.max_consistency_weight = 1.0
        self.rampup_length = 4000

    def train_step(self, batch):
        if len(batch) == 2:
            images, labels = batch
            unlabeled_images = None
        else:
            images, labels, unlabeled_images = batch

        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
    
        # Z-score normalization
        mean = tf.reduce_mean(images)
        std = tf.math.reduce_std(images) + 1e-6
        images = (images - mean) / std

        with tf.GradientTape() as tape:
            # Get predictions
            student_logits = self.student_model(images, training=True)
            
            if unlabeled_images is not None:
              # Calculate consistency loss only if unlabeled images are present
              teacher_logits = self.teacher_model(unlabeled_images, training=False)

            # Calculate supervised loss
            supervised_loss = self.supervised_loss_fn(labels, student_logits)

            # Calculate consistency weight with cosine rampup
            current_step = tf.cast(self.optimizer.iterations, tf.float32)
            rampup_ratio = tf.minimum(1.0, current_step / self.rampup_length)
            consistency_weight = self.max_consistency_weight * rampup_ratio  # Changed

            # MSE on probabilities
            if unlabeled_images is not None:
              teacher_probs = tf.nn.sigmoid(teacher_logits)
              student_probs = tf.nn.sigmoid(student_logits)
              consistency_loss = tf.reduce_mean(tf.square(teacher_probs - student_probs))
            else:
              consistency_loss = 0.0 # No consistency loss if no unlabeled images

            # Total loss with gradient scaling
            total_loss = supervised_loss + consistency_weight * consistency_loss

            # Scale loss for numerical stability
            scaled_loss = total_loss * 0.1

        # Compute and apply gradients with gradient accumulation
        gradients = tape.gradient(scaled_loss, self.student_model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]

        # Update student model
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model
        self._update_teacher()

        # Print debug info occasionally
        if self.optimizer.iterations % 100 == 0:
            tf.print("\nStep:", self.optimizer.iterations)
            tf.print("Learning rate:", self.optimizer.learning_rate)
            tf.print("Consistency weight:", consistency_weight)
            tf.print("Supervised loss:", supervised_loss)
            tf.print("Consistency loss:", consistency_loss)

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss
        }

    def _update_teacher(self):
        """Update teacher model with dynamic EMA"""
        # Ramp up EMA decay from 0.95 to 0.999
        current_step = tf.cast(self.optimizer.iterations, tf.float32)
        warmup_steps = 1000.0
        current_decay = 0.95 + 0.049 * tf.minimum(1.0, current_step / warmup_steps)
        
        # Debug print every 100 steps
        if current_step % 100 == 0:
            tf.print("\nCurrent EMA decay:", current_decay)
         # ADD THESE PRINT STATEMENTS
        tf.print("Student weights before update:", tf.reduce_mean(self.student_model.trainable_variables[0]))
        tf.print("Teacher weights before update:", tf.reduce_mean(self.teacher_model.trainable_variables[0]))


        for student_var, teacher_var in zip(
                self.student_model.trainable_variables,
                self.teacher_model.trainable_variables):
            delta = current_decay * (teacher_var - student_var)
            teacher_var.assign_sub(delta)
            
        # Verify update
        if current_step % 100 == 0:
            student_weights = self.student_model.get_weights()[0]
            teacher_weights = self.teacher_model.get_weights()[0]
            tf.print("Weight diff:", tf.reduce_mean(tf.abs(student_weights - teacher_weights)))

    def _compute_dice(self, y_true, y_pred):
        """Compute Dice score"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if len(y_true.shape) > 3:
            y_true = y_true[..., -1]
            y_pred = y_pred[..., -1]
        
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        sum_true = tf.reduce_sum(y_true, axis=[1, 2])
        sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
        
        dice = (2. * intersection + 1e-6) / (sum_true + sum_pred + 1e-6)
        dice = tf.where(tf.math.is_nan(dice), tf.zeros_like(dice), dice)
        
        return tf.reduce_mean(dice)

    def validate(self, val_dataset):
        """Run validation"""
        dice_scores = []
        teacher_dice_scores = []
        
        for batch in val_dataset:
            images, labels = batch
            
            student_logits = self.student_model(images, training=False)
            teacher_logits = self.teacher_model(images, training=False)
            
            student_dice = self._compute_dice(labels, student_logits)
            teacher_dice = self._compute_dice(labels, teacher_logits)
            
            if not tf.math.is_nan(student_dice) and not tf.math.is_nan(teacher_dice):
                dice_scores.append(float(student_dice))
                teacher_dice_scores.append(float(teacher_dice))
        
        if dice_scores and teacher_dice_scores:
            return np.mean(dice_scores), np.mean(teacher_dice_scores)
        return 0.0, 0.0

    def train(self, data_paths):
        """Main training loop with validation"""
        print("\nStarting training...")
        print("\nValidating data paths...")
        
        # Validate data paths before creating datasets
        for key, paths in data_paths['labeled'].items():
            print(f"\nChecking labeled {key}...")
            for path in paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Labeled {key} path does not exist: {path}")
                try:
                    data = np.load(path)
                    print(f"Successfully loaded {path}, shape: {data.shape}")
                except Exception as e:
                    raise ValueError(f"Error loading {path}: {e}")
        
        print("\nCreating datasets...")
        train_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['labeled']['images'],
            data_paths['labeled']['labels'],
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_ds = self.data_pipeline.dataloader.create_dataset(
            data_paths['validation']['images'],
            data_paths['validation']['labels'],
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Validate datasets
        try:
            print("\nValidating training dataset...")
            for i, batch in enumerate(train_ds.take(1)):
                print(f"Training batch shapes - Images: {batch[0].shape}, Labels: {batch[1].shape}")
                
            print("\nValidating validation dataset...")
            for i, batch in enumerate(val_ds.take(1)):
                print(f"Validation batch shapes - Images: {batch[0].shape}, Labels: {batch[1].shape}")
        except Exception as e:
            raise ValueError(f"Error validating datasets: {e}")
        
        print("\nStarting training loop...")
        
        best_dice = 0
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Training
            epoch_losses = []
            supervised_losses = []
            consistency_losses = []

            for batch in train_ds:
                metrics = self.train_step(batch)
                epoch_losses.append(float(metrics['total_loss']))
                supervised_losses.append(float(metrics['supervised_loss']))
                consistency_losses.append(float(metrics['consistency_loss']))

            # Validation
            val_dice, teacher_dice = self.validate(val_ds)

            # Update history
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['val_dice'].append(val_dice)
            self.history['teacher_dice'].append(teacher_dice)
            self.history['supervised_loss'].append(np.mean(supervised_losses))
            self.history['consistency_loss'].append(np.mean(consistency_losses))
            self.history['learning_rate'].append(
                float(self.lr_schedule(self.optimizer.iterations))
            )

            # Logging
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Time: {time.time() - epoch_start:.2f}s | "
                  f"Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"Teacher Dice: {teacher_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                self.save_checkpoint('best_model')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print("\nEarly stopping triggered!")
                break

            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress()

        print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        try:
            student_path = self.checkpoint_dir / f'{name}_student'
            teacher_path = self.checkpoint_dir / f'{name}_teacher'
            
            self.student_model.save_weights(str(student_path))
            self.teacher_model.save_weights(str(teacher_path))
            print(f"Saved checkpoint: {name}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def plot_progress(self):
        """Plot training progress"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Total Loss')
        ax.plot(self.history['supervised_loss'], label='Supervised Loss')
        ax.plot(self.history['consistency_loss'], label='Consistency Loss')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot Dice scores
        ax = axes[0, 1]
        ax.plot(self.history['val_dice'], label='Student Dice')
        ax.plot(self.history['teacher_dice'], label='Teacher Dice')
        ax.set_title('Validation Dice Scores')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.legend()
        
        # Plot learning rate
        ax = axes[1, 0]
        ax.plot(self.history['learning_rate'])
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'training_progress_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()

    def plot_final_results(self):
        """Create final training summary plots"""
        self.plot_progress()  # Save final progress plot
        
        # Save history to CSV
        import pandas as pd
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        print(f"Saved training history to {self.output_dir / 'training_history.csv'}")
#########################
#loss

# Add MeanTeacherTrainer class definition here

# In train_ssl_tf2n.py

class MeanTeacherTrainer(tf.keras.Model):
    def __init__(self, student_model, teacher_model, 
                 ema_decay, # This is the base_ema_decay for per-batch EMA
                 sharpening_temperature=0.5,
                 # This is passed from run_mean_teacher_v2.py args, used by test_step logic
                 teacher_direct_copy_epochs_config=0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        self.base_ema_decay = tf.cast(ema_decay, tf.float32)
        self.sharpening_temperature = tf.constant(sharpening_temperature, dtype=tf.float32)
        # Store the config for how many epochs test_step should force copy
        self.teacher_direct_copy_epochs_config = tf.constant(teacher_direct_copy_epochs_config, dtype=tf.int32)
        
        self.teacher_model.trainable = False
        self.consistency_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="consistency_weight")
        self.consistency_loss_fn = tf.keras.losses.MeanSquaredError()
        self.mt_phase_current_epoch = tf.Variable(-1, dtype=tf.int32, trainable=False, name="mt_phase_epoch_counter")
        #self.teacher_val_dice_metric = DiceCoefficient(name="teacher_val_dice_obj")
 
 
    # --- ADD THIS CALL METHOD ---
    def call(self, inputs, training=False, mask=None):
        """
        Defines the forward pass for the MeanTeacherTrainer model.
        For evaluation or direct calls, this will return the student model's output.
        """
        # When MeanTeacherTrainer itself is called (e.g., by some Keras internals
        # or if you did trainer_instance(inputs)), return the student's prediction.
        return self.student_model(inputs, training=training)
    # --- END ADDITION ---

    def _update_teacher_model(self): 
        # This method ALWAYS uses self.base_ema_decay for the per-batch EMA.
        # The TeacherDirectCopyCallback in run_mean_teacher_v2.py handles the initial direct copy at on_epoch_begin.
        for s_var, t_var in zip(self.student_model.trainable_variables, self.teacher_model.trainable_variables):
            t_var.assign(self.base_ema_decay * t_var + (1.0 - self.base_ema_decay) * s_var)
        
        student_all_weights = self.student_model.weights
        teacher_all_weights = self.teacher_model.weights
        if len(student_all_weights) == len(teacher_all_weights):
            for s_w, t_w in zip(student_all_weights, teacher_all_weights):
                if not t_w.trainable: 
                    if s_w.shape == t_w.shape: t_w.assign(s_w) 
        # else:
            # tf.print("ERROR _update_teacher_model BN: Student/Teacher weight count mismatch.", output_stream=sys.stderr)
            
    def _calculate_dice(self, y_true, y_pred_logits, smooth=1e-6):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_probs = tf.nn.sigmoid(y_pred_logits)
        y_pred_f_binary = tf.cast(y_pred_probs > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f_binary, axis=[1, 2, 3])
        sum_true = tf.reduce_sum(y_true_f, axis=[1, 2, 3])
        sum_pred = tf.reduce_sum(y_pred_f_binary, axis=[1, 2, 3])
        dice_per_sample = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
        return tf.reduce_mean(dice_per_sample)

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def train_step(self, data):
        labeled_data, unlabeled_data = data
        labeled_images, true_labels = labeled_data
        unlabeled_student_input, unlabeled_teacher_input = unlabeled_data

        with tf.GradientTape() as tape:
            student_labeled_logits = self.student_model(labeled_images, training=True)
            supervised_loss = self.compiled_loss(true_labels, student_labeled_logits, regularization_losses=self.student_model.losses)

            student_unlabeled_logits = self.student_model(unlabeled_student_input, training=True)
            teacher_unlabeled_logits = self.teacher_model(unlabeled_teacher_input, training=False)
            
            raw_sharpened_logits = teacher_unlabeled_logits / self.sharpening_temperature
            safe_sharpened_logits = tf.where(tf.math.is_finite(raw_sharpened_logits), raw_sharpened_logits, tf.zeros_like(raw_sharpened_logits))
            clipped_sharpened_logits = tf.clip_by_value(safe_sharpened_logits, -15.0, 15.0)
            sharpened_teacher_probs_for_consistency = tf.nn.sigmoid(clipped_sharpened_logits)

            student_unlabeled_probs = tf.nn.sigmoid(student_unlabeled_logits)
            consistency_loss_value = self.consistency_loss_fn(sharpened_teacher_probs_for_consistency, student_unlabeled_probs)
            total_loss = supervised_loss + self.consistency_weight * consistency_loss_value

        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        # Robust gradient clipping
        clipped_grads_and_vars = []
        for grad, var in zip(gradients, self.student_model.trainable_variables):
            if grad is not None:
                clipped_grads_and_vars.append((tf.clip_by_norm(grad, 1.0), var))
        self.optimizer.apply_gradients(clipped_grads_and_vars)

        self._update_teacher_model() 

        self.compiled_metrics.update_state(true_labels, student_labeled_logits)
        logs = {'loss': total_loss, 'supervised_loss': supervised_loss, 'consistency_loss': consistency_loss_value}
        for metric in self.metrics: logs[metric.name] = metric.result()
        return logs

    # --- THIS IS THE HELPER METHOD FOR EXPLICIT WEIGHT COPY ---
    def _perform_explicit_model_copy(self, source_model, target_model):
        """Helper function to explicitly copy weights layer by layer, component by component."""
        # tf.print(f"  ExplicitCopy: Attempting to copy from {source_model.name} to {target_model.name}", output_stream=sys.stderr)
        if len(source_model.layers) != len(target_model.layers):
            tf.print(f"  CRITICAL ExplicitCopy: Layer count MISMATCH! Source: {len(source_model.layers)}, Target: {len(target_model.layers)}", output_stream=sys.stderr)
            return False # Indicate failure

        all_layers_copied_successfully = True
        # Iterating through the top-level layers of the source and target models
        for s_top_layer, t_top_layer in zip(source_model.layers, target_model.layers):
            
            # Check if the current top-level layer is a UNetBlock instance
            if isinstance(s_top_layer, UNetBlock) and isinstance(t_top_layer, UNetBlock):
                try:
                    # tf.print(f"    Copying UNetBlock components: {s_top_layer.name} -> {t_top_layer.name}", output_stream=sys.stderr)
                    t_top_layer.conv1.set_weights(s_top_layer.conv1.get_weights())
                    t_top_layer.bn1.set_weights(s_top_layer.bn1.get_weights())
                    t_top_layer.conv2.set_weights(s_top_layer.conv2.get_weights())
                    t_top_layer.bn2.set_weights(s_top_layer.bn2.get_weights())
                    
                    # Dropout layers usually don't have weights, but this is safe if they did
                    if hasattr(s_top_layer, 'dropout') and s_top_layer.dropout is not None and \
                       hasattr(t_top_layer, 'dropout') and t_top_layer.dropout is not None and \
                       s_top_layer.dropout.weights: 
                        # Corrected: use t_top_layer and s_top_layer here
                        t_top_layer.dropout.set_weights(s_top_layer.dropout.get_weights())
                except Exception as e_ub_copy:
                    tf.print(f"    ERROR copying components of UNetBlock {s_top_layer.name}: {type(e_ub_copy).__name__} - {e_ub_copy}", output_stream=sys.stderr)
                    all_layers_copied_successfully = False
            
            # Handle other standard Keras layers that have weights (e.g., Conv2DTranspose, final Conv2D)
            elif hasattr(s_top_layer, 'weights') and s_top_layer.weights and \
                 hasattr(t_top_layer, 'set_weights') and callable(t_top_layer.set_weights):
                try:
                    # tf.print(f"    Copying generic Layer weights: {s_top_layer.name} -> {t_top_layer.name}", output_stream=sys.stderr)
                    t_top_layer.set_weights(s_top_layer.get_weights())
                except Exception as e_gen_copy:
                    tf.print(f"    ERROR copying weights for generic Layer {s_top_layer.name} (type {type(s_top_layer).__name__}): {type(e_gen_copy).__name__} - {e_gen_copy}", output_stream=sys.stderr)
                    all_layers_copied_successfully = False
            # else: # Layer has no weights (e.g., MaxPooling, ReLU as separate layer, Flatten, etc.)
                # tf.print(f"    Skipping Layer (no weights or not standard Keras layer): {s_top_layer.name}", output_stream=sys.stderr)
        
        return all_layers_copied_successfully
    # --- END HELPER METHOD ---
    @property
    def metrics(self):
        return super().metrics
    
    def test_step(self, data):
        val_images, val_labels = data
        # current_mt_epoch = self.mt_phase_current_epoch # Not strictly needed in this clean test_step

        # 1. Student Model Evaluation
        student_val_logits = self.student_model(val_images, training=False)
        # Calculate student's supervised loss on validation data
        # Keras automatically tracks this as 'val_loss' if 'loss' is the key from train_step.
        # The 'loss' key in the returned dict will be taken as the main validation loss.
        val_main_loss = self.compiled_loss(
            val_labels, 
            student_val_logits, 
            regularization_losses=self.student_model.losses
        )
        
        # Update student's compiled metrics (e.g., student_dice)
        # This will make self.metrics[i].result() have the validation value.
        self.compiled_metrics.update_state(val_labels, student_val_logits)

        # 2. Teacher Model Evaluation
        # Teacher model uses its current EMA'd weights.
        teacher_val_logits = self.teacher_model(val_images, training=False)
        # Calculate teacher's Dice score for THIS BATCH using the helper method
        teacher_dice_score_for_this_batch = self._calculate_dice(val_labels, teacher_val_logits) 
        
        # 3. Prepare Logs
        # Keras will average 'teacher_dice' (from per-batch returns) over all validation batches 
        # to get the epoch-level 'val_teacher_dice'.
        # The 'loss' key here will be logged by Keras as 'val_loss'.
        logs = {
            'loss': val_main_loss, 
            'teacher_dice': teacher_dice_score_for_this_batch
        }
        
        # Add results from student's compiled metrics
        # For a metric named 'student_dice', Keras will log its result as 'val_student_dice'.
        for metric_obj in self.metrics: 
            if metric_obj.name != 'loss': # Avoid re-adding 'loss' if it's in self.metrics by default
                 logs[metric_obj.name] = metric_obj.result() 
            
        return logs
    
    def _perform_explicit_model_copy(self, source_model, target_model):
        # ... (The robust layer-wise copy logic from previous responses) ...
        if len(source_model.layers) != len(target_model.layers): return False
        all_ok = True
        for s_top_layer, t_top_layer in zip(source_model.layers, target_model.layers):
            if isinstance(s_top_layer, UNetBlock) and isinstance(t_top_layer, UNetBlock):
                try:
                    t_top_layer.conv1.set_weights(s_top_layer.conv1.get_weights())
                    t_top_layer.bn1.set_weights(s_top_layer.bn1.get_weights())
                    t_top_layer.conv2.set_weights(s_top_layer.conv2.get_weights())
                    t_top_layer.bn2.set_weights(s_top_layer.bn2.get_weights())
                    if hasattr(s_top_layer,'dropout') and s_top_layer.dropout and hasattr(t_top_layer,'dropout') and t_top_layer.dropout and s_top_layer.dropout.weights:
                        t_top_layer.dropout.set_weights(s_top_layer.dropout.get_weights())
                except Exception as e: all_ok=False; tf.print(f"Err copy UNetBlock {s_top_layer.name}:{e}",sys.stderr)
            elif hasattr(s_top_layer,'weights') and s_top_layer.weights and hasattr(t_top_layer,'set_weights'):
                try: t_top_layer.set_weights(s_top_layer.get_weights())
                except Exception as e: all_ok=False; tf.print(f"Err copy Layer {s_top_layer.name}:{e}",sys.stderr)
        return all_ok    
    # @property
    # def metrics(self):
    #     # Override to include our custom teacher validation metric
    #     # This ensures Keras handles its state (e.g., reset_state at epoch start)
    #     base_metrics = super().metrics
    #     return base_metrics + [self.teacher_val_dice_metric]

class StableConsistencyLoss(tf.keras.losses.Loss):
    """Stable Consistency Loss with temperature scaling"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def call(self, teacher_logits, student_logits):
        # Apply temperature scaling
        teacher_probs = tf.nn.sigmoid(teacher_logits / self.temperature)
        student_probs = tf.nn.sigmoid(student_logits / self.temperature)
        
        # Calculate MSE loss
        return tf.reduce_mean(tf.square(teacher_probs - student_probs))

class CombinedSSLLoss(tf.keras.losses.Loss):
    """Combined loss for semi-supervised learning"""
    def __init__(self, dice_weight=1.0, consistency_weight=1.0):
        super().__init__()
        self.dice_loss = StableDiceLoss()
        self.consistency_loss = StableConsistencyLoss()
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
    
    def call(self, y_true, y_pred, teacher_pred=None):
        losses = {'dice_loss': self.dice_loss(y_true, y_pred)}
        
        if teacher_pred is not None:
            losses['consistency_loss'] = self.consistency_loss(teacher_pred, y_pred)
            total_loss = (self.dice_weight * losses['dice_loss'] + 
                         self.consistency_weight * losses['consistency_loss'])
        else:
            total_loss = losses['dice_loss']
        
        return total_loss
#########################
# In train_ssl_tf2n.py, modify/replace the MixMatchTrainer:

class MixMatchTrainer:
    def __init__(self, config): # Expects StableSSLConfig or similar
        tf.print("Initializing MixMatch Trainer...")
        self.config = config
        # Ensure DataPipeline is correctly imported or defined if PancreasDataLoader is separate
        self.data_pipeline = DataPipeline(config) 

        # MixMatch hyperparameters from config or defaults
        self.T = getattr(config, 'mixmatch_T', 0.5)
        self.K = getattr(config, 'mixmatch_K', 2)
        self.alpha_mixup = getattr(config, 'mixmatch_alpha', 0.75) # MixUp alpha
        self.lambda_u_max = getattr(config, 'mixmatch_consistency_max', 75.0) # Max weight for unlabeled loss
        self.lambda_u_rampup_steps = getattr(config, 'mixmatch_rampup_steps', 500) 

        # Output directories
        self.output_dir_base = Path(getattr(config, 'output_dir', 'mixmatch_results_default'))
        self.experiment_name = getattr(config, 'experiment_name', f"MixMatch_Exp_{time.strftime('%Y%m%d_%H%M%S')}")
        self.output_dir = self.output_dir_base / self.experiment_name
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plots_dir = self.output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        tf.print(f"MixMatch results, checkpoints, plots will be in: {self.output_dir}")

        # Model
        tf.print("Creating MixMatch student model...")
        self.model = PancreasSeg(config) 
        dummy_input_shape = (1, config.img_size_y, config.img_size_x, config.num_channels)
        try:
            _ = self.model(tf.zeros(dummy_input_shape), training=False)
            tf.print(f"MixMatch student model '{self.model.name}' built.")
        except Exception as e:
            tf.print(f"ERROR building MixMatch student model: {e}", output_stream=sys.stderr); raise e

        # Optimizer and Learning Rate Schedule
        initial_lr_val = getattr(config, 'learning_rate', 0.002) 
        decay_steps_val = getattr(config, 'lr_decay_steps', 1000) # Example: steps for one decay cycle

        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr_val, 
            first_decay_steps=decay_steps_val, 
            t_mul=2.0, m_mul=0.95, alpha=0.01 # alpha is min_lr_factor
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule) # Assign schedule
        tf.print(f"MixMatch Optimizer: Adam with CosineDecayRestarts LR schedule (initial: {initial_lr_val}).")
        
        # Loss Functions
        #self.supervised_loss_fn = CombinedLoss(config) # Assumes CombinedLoss uses Dice+Focal

        self.supervised_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.unsupervised_loss_fn = tf.keras.losses.MeanSquaredError()
        tf.print(f"  Unsupervised loss (L_u) set to: {type(self.unsupervised_loss_fn).__name__}")
        self.history = {
            'epoch': [], 'total_loss': [], 'L_x': [], 'L_u': [], 'lambda_u': [],
            'val_dice': [], 'learning_rate': []
        }
        tf.print("MixMatch trainer initialization complete! Using BinaryCrossentropy for L_x.")         
        tf.print("MixMatch trainer initialization complete!")

    def _sharpen_probs(self, p, T):
        if T == 0: return tf.cast(p > 0.5, p.dtype)
        p_pow_t = tf.pow(p, 1.0/T)
        one_minus_p_pow_t = tf.pow(1.0 - p, 1.0/T)
        return p_pow_t / (p_pow_t + one_minus_p_pow_t + 1e-7)

    def _mixup(self, x1, x2, y1, y2, alpha):
        batch_size = tf.shape(x1)[0]
        # Use tfp.distributions.Beta if available and preferred, otherwise tf.compat.v1 works
        try:
            beta_dist =  tf.compat.v1.distributions.Beta(alpha, alpha)
        except ImportError:
            tf.print("WARNING: tensorflow_probability not found, using tf.compat.v1.distributions.Beta for MixUp.", output_stream=sys.stderr)
            beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
            
        lambda_ = beta_dist.sample(batch_size)
        lambda_ = tf.maximum(lambda_, 1.0 - lambda_)
        lambda_x_reshaped = tf.reshape(lambda_, [batch_size, 1, 1, 1])
        x_mixed = lambda_x_reshaped * x1 + (1.0 - lambda_x_reshaped) * x2
        lambda_y_reshaped = tf.reshape(lambda_, [batch_size, 1, 1, 1])
        y_mixed = lambda_y_reshaped * y1 + (1.0 - lambda_y_reshaped) * y2
        return x_mixed, y_mixed, lambda_

    def _get_lambda_u(self, step):
        return self.lambda_u_max * tf.minimum(1.0, tf.cast(step, tf.float32) / tf.cast(self.lambda_u_rampup_steps, tf.float32))

    @tf.function
    def train_step(self, labeled_batch, unlabeled_batch_raw):
        images_xl, labels_pl = labeled_batch 
        images_xu_raw = unlabeled_batch_raw[0] 

        guessed_labels_qb_list = []
        for _ in range(self.K):
            xu_k_aug = tf.map_fn(lambda x: self.data_pipeline.dataloader._augment_single_image_slice(x, strength='weak'), images_xu_raw)
            logits_xu_k = self.model(xu_k_aug, training=False)
            probs_xu_k = tf.nn.sigmoid(logits_xu_k)
            guessed_labels_qb_list.append(probs_xu_k)
        
        avg_guessed_probs_qb = tf.reduce_mean(tf.stack(guessed_labels_qb_list), axis=0)
        sharpened_pseudo_labels_qu = self._sharpen_probs(avg_guessed_probs_qb, self.T)

        images_xl_aug = tf.map_fn(lambda x: self.data_pipeline.dataloader._augment_single_image_slice(x, strength='strong'), images_xl)
        images_xu_aug = tf.map_fn(lambda x: self.data_pipeline.dataloader._augment_single_image_slice(x, strength='strong'), images_xu_raw)

        bs_l = tf.shape(images_xl_aug)[0]
        shuffle_l_indices = tf.random.shuffle(tf.range(bs_l))
        X_l_mixed, P_l_mixed, _ = self._mixup(
            images_xl_aug, tf.gather(images_xl_aug, shuffle_l_indices),
            labels_pl, tf.gather(labels_pl, shuffle_l_indices), self.alpha_mixup)

        bs_u = tf.shape(images_xu_aug)[0]
        shuffle_u_indices = tf.random.shuffle(tf.range(bs_u))
        X_u_mixed, Q_u_mixed, _ = self._mixup(
            images_xu_aug, tf.gather(images_xu_aug, shuffle_u_indices),
            sharpened_pseudo_labels_qu, tf.gather(sharpened_pseudo_labels_qu, shuffle_u_indices), self.alpha_mixup)
            
        with tf.GradientTape() as tape:
            logits_Lx_student = self.model(X_l_mixed, training=True)
            loss_Lx = self.supervised_loss_fn(P_l_mixed, logits_Lx_student) 

            logits_Lu_student = self.model(X_u_mixed, training=True)
            probs_Lu_student = tf.nn.sigmoid(logits_Lu_student)
            loss_Lu = self.unsupervised_loss_fn(Q_u_mixed, probs_Lu_student)

            lambda_u_weight = self._get_lambda_u(self.optimizer.iterations)
            total_loss = loss_Lx + lambda_u_weight * loss_Lu
        
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        clipped_grads = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in gradients]
        grads_and_vars_final = [(g,v) for g,v in zip(clipped_grads, trainable_vars) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars_final)

        return {'total_loss': total_loss, 'L_x': loss_Lx, 'L_u': loss_Lu, 'lambda_u': lambda_u_weight}
    
    def _compute_dice_metric(self, y_true, y_pred_logits):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_probs = tf.nn.sigmoid(y_pred_logits)
        y_pred_f_binary = tf.cast(y_pred_probs > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true_f * y_pred_f_binary, axis=[1,2,3])
        union_sum = tf.reduce_sum(y_true_f, axis=[1,2,3]) + tf.reduce_sum(y_pred_f_binary, axis=[1,2,3])
        dice_per_sample = (2. * intersection + 1e-6) / (union_sum + 1e-6)
        return tf.reduce_mean(dice_per_sample)

    def validate(self, val_dataset):
        dice_scores = []; total_loss = []; count = 0
        if val_dataset is None: return 0.0, 0.0 # Return avg_loss, avg_dice
        for images, labels in val_dataset:
            logits = self.model(images, training=False)
            # For validation loss, use the supervised loss component
            loss = self.supervised_loss_fn(labels, logits) 
            dice = self._compute_dice_metric(labels, logits)
            if not tf.math.is_nan(dice): dice_scores.append(dice.numpy())
            if not tf.math.is_nan(loss): total_loss.append(loss.numpy())
            count +=1
        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        avg_loss = np.mean(total_loss) if total_loss else 0.0 # Can add val_loss to history
        return avg_dice # Only returning dice for now to match existing history structure

    def train(self, data_paths, num_epochs, steps_per_epoch=None):
        tf.print(f"\nStarting MixMatch training for {num_epochs} epochs...")
        train_ds_l = self.data_pipeline.build_labeled_dataset(data_paths['labeled']['images'], data_paths['labeled']['labels'], self.config.batch_size, True).prefetch(AUTOTUNE)
        train_ds_u_raw = self.data_pipeline.build_unlabeled_dataset_for_mixmatch(data_paths['unlabeled']['images'], self.config.batch_size, True).repeat().prefetch(AUTOTUNE)
        val_ds = self.data_pipeline.build_validation_dataset(data_paths['validation']['images'], data_paths['validation']['labels'], self.config.batch_size).prefetch(AUTOTUNE)

        if steps_per_epoch is None:
            num_labeled_files = len(data_paths['labeled']['images']); num_labeled_samples = num_labeled_files 
            if num_labeled_samples > 0: steps_per_epoch = (num_labeled_samples + self.config.batch_size -1)//self.config.batch_size
            else: steps_per_epoch = 100 
            tf.print(f"MixMatch using steps_per_epoch: {steps_per_epoch}")

        train_iter_l = iter(train_ds_l.repeat()) # Repeat labeled too, to ensure enough data for steps_per_epoch
        train_iter_u_raw = iter(train_ds_u_raw)
        
        best_val_dice = 0.0; patience_counter = 0
        early_stopping_patience = getattr(self.config, 'early_stopping_patience', 20)
        csv_log_path = self.output_dir / "mixmatch_training_log.csv"
        with open(csv_log_path, 'w') as f: f.write("epoch,total_loss,L_x,L_u,lambda_u,val_dice,learning_rate,time_epoch_s\n")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_total_L, epoch_Lx, epoch_Lu, epoch_lambda_u = [], [], [], []
            progbar = tf.keras.utils.Progbar(steps_per_epoch, unit_name='step', stateful_metrics=['total_L','L_x','L_u','lambda_u'])
            for step in range(steps_per_epoch):
                labeled_batch = next(train_iter_l)
                unlabeled_batch_raw = next(train_iter_u_raw)
                metrics = self.train_step(labeled_batch, unlabeled_batch_raw)
                epoch_total_L.append(metrics['total_loss'].numpy()); epoch_Lx.append(metrics['L_x'].numpy())
                epoch_Lu.append(metrics['L_u'].numpy()); epoch_lambda_u.append(metrics['lambda_u'].numpy())
                progbar.update(step + 1, values=[("T_L", metrics['total_loss']), ("L_x", metrics['L_x']), ("L_u", metrics['L_u']), ("l_u", metrics['lambda_u'])])

            val_dice_epoch = self.validate(val_ds)
            #current_lr_epoch = self.lr_schedule(self.optimizer.iterations).numpy()
            current_lr_val = self.lr_schedule(self.optimizer.iterations)
            current_lr_epoch = current_lr_val.numpy()
            self.history['epoch'].append(epoch + 1); self.history['total_loss'].append(np.mean(epoch_total_L))
            self.history['L_x'].append(np.mean(epoch_Lx)); self.history['L_u'].append(np.mean(epoch_Lu))
            self.history['lambda_u'].append(np.mean(epoch_lambda_u)); self.history['val_dice'].append(val_dice_epoch)
            self.history['learning_rate'].append(current_lr_epoch)
            
            epoch_time = time.time() - epoch_start_time
            tf.print(f"\nE{epoch+1}/{num_epochs}-T:{epoch_time:.1f}s-LR:{current_lr_epoch:.2e} Loss:T={self.history['total_loss'][-1]:.3f},X={self.history['L_x'][-1]:.3f},U={self.history['L_u'][-1]:.3f}(w={self.history['lambda_u'][-1]:.2f}) ValDice:{val_dice_epoch:.4f}")
            with open(csv_log_path, 'a') as f: f.write(f"{epoch+1},{self.history['total_loss'][-1]:.6f},{self.history['L_x'][-1]:.6f},{self.history['L_u'][-1]:.6f},{self.history['lambda_u'][-1]:.4f},{val_dice_epoch:.6f},{current_lr_epoch:.8e},{epoch_time:.2f}\n")

            if val_dice_epoch > best_val_dice:
                best_val_dice = val_dice_epoch; self.save_checkpoint('best_mixmatch_model_student'); 
                tf.print(f"  ✓ Best val Dice: {best_val_dice:.4f}. Checkpoint saved."); patience_counter = 0
            else: patience_counter += 1
            
            if (epoch+1)%5==0 or epoch==num_epochs-1 or patience_counter>=early_stopping_patience: self.plot_progress(epoch+1, final_plot=(epoch==num_epochs-1 or patience_counter>=early_stopping_patience))
            if patience_counter >= early_stopping_patience: tf.print(f"\nEarly stopping at epoch {epoch+1}."); break
        
        tf.print(f"\nMixMatch Training complete! Best val Dice: {best_val_dice:.4f}")
        self.save_checkpoint('final_mixmatch_model_student')
        return self.history

    def save_checkpoint(self, name_prefix):
        model_path = self.checkpoint_dir / f'{name_prefix}_{time.strftime("%Y%m%d_%H%M%S")}.weights.h5'
        self.model.save_weights(str(model_path))
        tf.print(f"Saved MixMatch model checkpoint to {model_path}")

    def plot_progress(self, current_epoch_num, final_plot=False):
        # ... (Plotting logic from before, ensure it uses self.history keys correctly) ...
        if not self.history['epoch']: return
        plt.style.use('seaborn-v0_8-whitegrid'); fig, axes = plt.subplots(2,2,figsize=(18,10))
        fig.suptitle(f'MixMatch: {self.experiment_name} - Epoch {current_epoch_num}', fontsize=16)
        ep = self.history['epoch']
        ax=axes[0,0]; ax.plot(ep,self.history['total_loss'],label='Total',c='r'); ax.plot(ep,self.history['L_x'],label='L_x',c='b',ls='--'); ax.plot(ep,self.history['L_u'],label='L_u',c='g',ls=':'); ax.set_title('Losses'); ax.legend(); ax.grid(True)
        ax=axes[0,1]; ax.plot(ep,self.history['val_dice'],label='Val Dice',c='purple'); ax.set_title('Val Dice'); ax.set_ylim(0,1.05); ax.legend(); ax.grid(True)
        ax=axes[1,0]; ax.plot(ep,self.history['lambda_u'],label='$\lambda_u$',c='brown'); ax.set_title('Cons. Weight'); ax.legend(); ax.grid(True)
        ax=axes[1,1]; ax.plot(ep,self.history['learning_rate'],label='LR',c='teal'); ax.set_title('Learning Rate'); ax.set_yscale('log'); ax.legend(); ax.grid(True)
        plt.tight_layout(rect=[0,0,1,0.95])
        fn=f"{'final' if final_plot else f'epoch_{current_epoch_num}'}_mixmatch_curves.png"
        plt.savefig(self.plots_dir/fn); plt.close(fig); tf.print(f"Saved plot: {self.plots_dir/fn}")
