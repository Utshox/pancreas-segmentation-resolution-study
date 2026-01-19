
"""
Phase 9: Patch-Based Mean Teacher (SSL)
---------------------------------------
Implements Mean Teacher on 256x256 high-res patches.
1. Student Model: Standard U-Net, updated by Gradient Descent.
2. Teacher Model: EMA (Exponential Moving Average) of Student weights.
3. Consistency Loss: MSE(Student(x), Teacher(x)).
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
BATCH_SIZE = 8 # (4 labeled + 4 unlabeled)
EPOCHS = 100
LR = 1e-4
ALPHA = 0.999 # EMA decay for teacher
LAMBDA_U = 10.0 # Weight for consistency loss (Needs rampup usually, but static is fine for pilot)

# --- U-Net ---
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

# --- Data Loading ---
def load_dataset_into_ram(data_dir):
    print("Loading Dataset into RAM...")
    x_files = sorted(list(Path(data_dir).glob('*_x.npy')))
    y_files = sorted(list(Path(data_dir).glob('*_y.npy')))
    
    all_x = []
    all_y = []
    
    for xf, yf in zip(x_files, y_files):
        try:
            bx = np.load(xf)
            by = np.load(yf)
            if len(bx) > 0:
                all_x.append(bx)
                all_y.append(by)
        except:
            pass
            
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    
    X = np.expand_dims(X, axis=-1).astype(np.float32)
    Y = np.expand_dims(Y, axis=-1).astype(np.float32)
    Y = np.clip(Y, 0, 1) # Force binary
    
    print(f"Dataset Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y

# --- Mean Teacher Trainer ---
class MeanTeacherTrainer(models.Model):
    def __init__(self, student_model, teacher_model, alpha=0.999, lambda_u=10.0):
        super(MeanTeacherTrainer, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mse = tf.keras.losses.MeanSquaredError()

    def compile(self, optimizer, metrics):
        super(MeanTeacherTrainer, self).compile(optimizer=optimizer, metrics=metrics)

    def update_teacher(self):
        # Update All Variables (Weights + BN stats) using Assign (Graph-safe)
        for t_var, s_var in zip(self.teacher.variables, self.student.variables):
            t_var.assign(self.alpha * t_var + (1.0 - self.alpha) * s_var)

    def train_step(self, data):
        # data: ((x_l, y_l), x_u)
        (x_l, y_l), x_u = data
        
        # Concatenate for student pass? Or separate? 
        # Mean Teacher often passes x_l and x_u through student, and x_u through teacher.
        # Augmentation: Teacher input usually has weak noise (or none), Student has strong noise.
        # For simplicity, we assume data generator gives raw images, 
        # and we apply noise (dropout) internally if model has it.
        # But our unet doesn't have dropout.
        # We add Gaussian noise to student input to create difference.
        
        noise = tf.random.normal(shape=tf.shape(x_u), mean=0.0, stddev=0.05)
        x_u_student = x_u + noise
        x_u_teacher = x_u # Cleaner input for teacher
        
        with tf.GradientTape() as tape:
            # 1. Supervised Loss (Student on Labeled)
            logits_l_student = self.student(x_l, training=True)
            loss_sup = self.bce(y_l, logits_l_student)
            
            # 2. Consistency Loss (Unlabeled)
            logits_u_student = self.student(x_u_student, training=True)
            logits_u_teacher = self.teacher(x_u_teacher, training=False)
            
            # Simple MSE between predictions (probabilities)
            loss_con = self.mse(logits_u_teacher, logits_u_student)
            
            total_loss = loss_sup + self.lambda_u * loss_con

        # Update Student
        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        
        # Update Teacher (EMA)
        self.update_teacher()

        self.compiled_metrics.update_state(y_l, logits_l_student)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss_sup, "loss_con": loss_con})
        return results

    def test_step(self, data):
        x, y = data
        # Evaluate Student or Teacher? Ideally Teacher is better.
        y_pred = self.teacher(x, training=False)
        loss = self.bce(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# --- Generator (Same as FixMatch) ---
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
        
    def __len__(self):
        return int(self.n_labeled // (self.batch_size // 2))

    def __getitem__(self, index):
        bs_l = self.batch_size // 2
        inds_l = np.random.choice(self.indices_l, bs_l)
        X_l_batch = self.x_l[inds_l]
        Y_l_batch = self.y_l[inds_l]
        
        inds_u = np.random.choice(self.indices_u, bs_l)
        X_u_batch = self.x_u[inds_u]
        
        return ((X_l_batch, Y_l_batch), X_u_batch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_meanteacher')
    parser.add_argument('--labeled_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    # Load Data
    X, Y = load_dataset_into_ram(args.data_dir)
    
    n_total = len(X)
    n_labeled = int(n_total * args.labeled_ratio)
    p = np.random.permutation(n_total)
    X, Y = X[p], Y[p]
    
    val_split = int(0.1 * n_total)
    X_val = X[-val_split:]
    Y_val = Y[-val_split:]
    X_train = X[:-val_split]
    Y_train = Y[:-val_split]
    
    n_train = len(X_train)
    n_lbl = int(n_train * args.labeled_ratio)
    
    X_l = X_train[:n_lbl]
    Y_l = Y_train[:n_lbl]
    X_u = X_train[n_lbl:]
    
    print(f"Mean Teacher Config: Labeled={len(X_l)}, Unlabeled={len(X_u)}")
    
    train_gen = DualDataGenerator(X_l, Y_l, X_u, batch_size=BATCH_SIZE)
    
    # Models
    student = get_unet()
    teacher = get_unet()
    teacher.set_weights(student.get_weights()) # Initialize identically
    
    mt_model = MeanTeacherTrainer(student, teacher, alpha=ALPHA, lambda_u=LAMBDA_U)
    
    mt_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        metrics=['accuracy', tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='io_u')]
    )
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Note: We save weights from the wrapper. 
    # Importantly: we ideally want the Teacher weights.
    # The ModelCheckpoint on 'val_io_u' calls test_step, which uses Teacher.
    # So saving 'model' saves the whole wrapper. 
    # Inference script will need to be adapted to load wrapper and extract teacher.
    
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model_mt_best.h5'), 
                                 save_best_only=True, monitor='val_io_u', mode='max', save_weights_only=True)
    
    mt_model.fit(
        train_gen,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        callbacks=[checkpoint, CSVLogger(os.path.join(args.output_dir, 'log.csv'))]
    )

if __name__ == "__main__":
    main()
