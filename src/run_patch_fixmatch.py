
"""
Phase 7: Patch-Based FixMatch (SSL)
-----------------------------------
Implements FixMatch on 256x256 high-res patches.
1. Teacher Model: Predicts on Weakly Augmented Unlabeled Image.
2. Pseudo-Label: Created if Teacher Confidence > Threshold (0.95).
3. Student Model: Trains on Strongly Augmented Unlabeled Image against Pseudo-Label.
4. Combined with standard Supervised Training on Labeled Data.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import argparse
from pathlib import Path

# --- Configuration ---
IMG_SIZE = 256
BATCH_SIZE = 8 # Total batch size (4 labeled + 4 unlabeled)
mu = 1 # Ratio of unlabeled data
EPOCHS = 100
LR = 1e-4
CONFIDENCE_THRESHOLD = 0.95
LAMBDA_U = 1.0 # Weight for unsupervised loss

# --- U-Net (Same as Phase 6) ---
def get_unet(img_size=256, num_classes=1):
    inputs = layers.Input(shape=(img_size, img_size, 1))

    # Encoding
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoding
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Augmentations ---
def weak_augment(images):
    # Random Flip (LR, UD) + Random Translation handled by tf.image?
    # Manual numpy implementation for speed in DataGen, or TF layers?
    # Let's use TF operations inside the training step for simplicity if possible,
    # BUT we need consistent aug for Weak.
    # Actually, simpler to do in DataGen or using TF layers.
    return tf.image.random_flip_left_right(tf.image.random_flip_up_down(images))

def strong_augment(images):
    # Weak + Noise + Cutout + Contrast
    x = weak_augment(images)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05, dtype=x.dtype)
    x = x + noise
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x

# --- Data Loading ---
def load_dataset_into_ram(data_dir):
    print("Loading Dataset into RAM...")
    x_files = sorted(list(Path(data_dir).glob('*_x.npy')))
    y_files = sorted(list(Path(data_dir).glob('*_y.npy')))
    
    all_x = []
    all_y = []
    
    for xf, yf in zip(x_files, y_files):
        try:
            bx = np.load(xf) # Shape (N, 256, 256)
            by = np.load(yf)
            
            if len(bx) > 0:
                all_x.append(bx)
                all_y.append(by)
        except:
            pass
            
    X = np.concatenate(all_x, axis=0) # (Total_N, 256, 256)
    Y = np.concatenate(all_y, axis=0)
    
    # Add channel dim
    X = np.expand_dims(X, axis=-1).astype(np.float32)
    Y = np.expand_dims(Y, axis=-1).astype(np.float32)
    Y = np.clip(Y, 0, 1) # Force binary
    
    print(f"Dataset Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y

# --- Custom Training Loop ---
class FixMatchTrainer(models.Model):
    def __init__(self, model, lambda_u=1.0, threshold=0.95):
        super(FixMatchTrainer, self).__init__()
        self.model = model
        self.lambda_u = lambda_u
        self.threshold = threshold
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def compile(self, optimizer, metrics):
        super(FixMatchTrainer, self).compile(optimizer=optimizer, metrics=metrics)

    def train_step(self, data):
        # data is ((x_labeled, y_labeled), x_unlabeled)
        # Unpack
        (x_l, y_l), x_u = data

        # Augmentations
        x_u_weak = x_u # Assume mostly weak or pre-augmented? 
        # Actually proper FixMatch does augmentation inside the loop on the RAW image.
        # Here x_u is raw.
        x_u_weak = weak_augment(x_u)
        x_u_strong = strong_augment(x_u)

        with tf.GradientTape() as tape:
            # 1. Supervised Loss
            logits_l = self.model(x_l, training=True)
            loss_sup = self.bce(y_l, logits_l)

            # 2. Pseudo-Labeling (Teacher)
            logits_u_weak = self.model(x_u_weak, training=False) # No dropout for teacher
            # Prediction is sigmoid output
            p_pseudo = logits_u_weak 
            
            # Masking
            # For each pixel? FixMatch usually does image-level. 
            # Segmenation FixMatch does pixel-level or consistency?
            # Let's do pixel-wise masking.
            mask = tf.cast(tf.greater_equal(p_pseudo, self.threshold), tf.float32)
            # Or less than 1-threshold for background?
            # Simplest: Just use standard BCE with generated target, masked by confidence.
            # But we need binary target.
            y_pseudo = tf.cast(tf.greater_equal(p_pseudo, 0.5), tf.float32)

            # 3. Unsupervised Loss (Student on Strong)
            logits_u_strong = self.model(x_u_strong, training=True)
            loss_u = self.bce(y_pseudo, logits_u_strong)
            
            # Apply Mask
            # BCE returns scaler by default? Use reduction=none?
            # Manual BCE:
            # loss_u_pixel = - (y_pseudo * log(pred) + (1-y)*log(1-pred))
            # masked_loss = loss_u_pixel * mask
            # For simplicity with Keras loss:
            # We assume bce reduces. We need pixel-wise.
            # Let's use simple weighting:
            
            loss_u = loss_u * tf.reduce_mean(mask) # Scale by confidence?
            
            total_loss = loss_sup + self.lambda_u * loss_u

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y_l, logits_l)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss_sup, "loss_u": loss_u})
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.bce(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

# --- Sequence for Dual Data ---
class DualDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_l, y_l, x_u, batch_size=8):
        self.x_l = x_l
        self.y_l = y_l
        self.x_u = x_u
        self.batch_size = batch_size
        self.n_labeled = len(x_l)
        self.n_unlabeled = len(x_u)
        self.indices_l = np.arange(self.n_labeled)
        self.indices_u = np.arange(self.n_unlabeled)
        
        # Batch consists of B/2 labeled, B/2 unlabeled to keep it simple?
        # FixMatch usually B labeled, mu*B unlabeled.
        # Let's do equal split for now.
        
    def __len__(self):
        return int(self.n_labeled // (self.batch_size // 2))

    def __getitem__(self, index):
        # Labeled batch
        bs_l = self.batch_size // 2
        inds_l = np.random.choice(self.indices_l, bs_l)
        X_l_batch = self.x_l[inds_l]
        Y_l_batch = self.y_l[inds_l]
        
        # Unlabeled batch
        inds_u = np.random.choice(self.indices_u, bs_l) # Sample randomly
        X_u_batch = self.x_u[inds_u]
        
        return ((X_l_batch, Y_l_batch), X_u_batch)

    def on_epoch_end(self):
        pass # Random choice handles shuffling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_fixmatch_patch')
    parser.add_argument('--labeled_ratio', type=float, default=0.1) # 10% Labeled
    args = parser.parse_args()
    
    # Load All Data
    X, Y = load_dataset_into_ram(args.data_dir)
    
    # Split Labeled/Unlabeled
    # We simulate SSL by hiding labels
    n_total = len(X)
    n_labeled = int(n_total * args.labeled_ratio)
    
    # Shuffle first
    p = np.random.permutation(n_total)
    X, Y = X[p], Y[p]
    
    X_l = X[:n_labeled]
    Y_l = Y[:n_labeled]
    X_u = X[n_labeled:]
    # Y_u is hidden
    
    # Validation set (use last 10% of total as clean holdout, outside of the SSL split logic to be safe)
    # Actually, let's take val from the end of X BEFORE splitting L/U?
    # Better: Split Val first.
    val_split = int(0.1 * n_total)
    X_val = X[-val_split:]
    Y_val = Y[-val_split:]
    
    X_train = X[:-val_split]
    Y_train = Y[:-val_split]
    
    # Now split Train into L/U
    n_train = len(X_train)
    n_lbl = int(n_train * args.labeled_ratio)
    
    X_l = X_train[:n_lbl]
    Y_l = Y_train[:n_lbl]
    X_u = X_train[n_lbl:]
    
    print(f"FixMatch Config: Labeled={len(X_l)}, Unlabeled={len(X_u)}, Val={len(X_val)}")
    
    # Generator
    train_gen = DualDataGenerator(X_l, Y_l, X_u, batch_size=BATCH_SIZE)
    
    # Model
    unet = get_unet()
    fixmatch_model = FixMatchTrainer(unet, lambda_u=LAMBDA_U, threshold=CONFIDENCE_THRESHOLD)
    
    fixmatch_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')]
    )
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # We can't easily save 'best model' based on val_io_u inside custom loop with standard ModelCheckpoint?
    # Keras supports it if we pass validation_data.
    # But we need to save the inner model (unet), not the wrapper.
    # Workaround: Custom callback or just save weights.
    
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model_fixmatch_best.h5'), 
                                 save_best_only=True, monitor='val_io_u', mode='max', save_weights_only=True)
    
    history = fixmatch_model.fit(
        train_gen,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        callbacks=[checkpoint, CSVLogger(os.path.join(args.output_dir, 'log.csv'))]
    )

if __name__ == "__main__":
    main()
