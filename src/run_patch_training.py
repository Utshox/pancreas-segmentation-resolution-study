
"""
Phase 6: Patch-Based U-Net Training
-----------------------------------
This script implements the "Stroke Paper" approach:
1. Loads 256x256 patches from `preprocessed_v5_patches/`.
2. Trains a standard U-Net (no resizing).
3. Validates on full images using Sliding Window Inference (optional, or separate script).
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import argparse
from pathlib import Path

# --- Configuration ---
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

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

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6-1 if False else c7) # Typo fix in logic, used c7
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

# --- Data Generator ---
class PatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_files, y_files, batch_size=32, shuffle=True, augment=False):
        self.x_files = x_files
        self.y_files = y_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.x_files))
        self.on_epoch_end()

        # Load all patches into memory if possible? 
        # N ~ 44,000 * 256KB = 11GB. Might get tight on 32GB RAM.
        # Strategy: Keep files, load on fly. 'x_files' is list of .npy paths.
        # WAIT! preprocessed_v5 splits by CASE. 
        # We need to flatten the list of matches.
        
    def __len__(self):
        # We don't know total patches yet if we just have case files.
        # Modified logic: Load one huge list of (case_id, patch_idx) is too slow.
        # Better: Pre-load dataset into huge memmap or HDF5.
        # Fallback for now: Load ALL .npy files into RAM. 11GB is fine for 32GB node.
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Placeholder for RAM based
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 1))
        y = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 1))
        
        for i, idx in enumerate(indexes):
            # If using RAM cache:
            X[i] = self.x_data[idx]
            y_raw = self.y_data[idx]
            y[i] = np.clip(y_raw, 0, 1) # Force binary labels
            
            if self.augment:
                # Simple augmentations
                if np.random.rand() > 0.5: # Flip LR
                    X[i] = np.fliplr(X[i])
                    y[i] = np.fliplr(y[i])
                if np.random.rand() > 0.5: # Flip UD
                    X[i] = np.flipud(X[i])
                    y[i] = np.flipud(y[i])
                    
        return X, y

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
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)
    Y = np.clip(Y, 0, 1) # Force binary labels (0, 1) to prevent IoU crash
    
    print(f"Dataset Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_v5')
    args = parser.parse_args()
    
    # Load Data
    X, Y = load_dataset_into_ram(args.data_dir)
    
    # Split
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    # Model
    model = get_unet()
    model.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss='binary_crossentropy', # Start simple, maybe dice later
                  metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')])
                  
    # Callbacks
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model_patch_best.h5'), 
                                 save_best_only=True, monitor='val_io_u', mode='max')
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, CSVLogger(os.path.join(args.output_dir, 'log.csv'))]
    )

if __name__ == "__main__":
    main()
