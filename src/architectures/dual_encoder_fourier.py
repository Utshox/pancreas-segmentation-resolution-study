"""
Dual-Encoder with Fourier Transform Branch
Architecture as suggested by Kursat:
- Branch A: CNN encoder on spatial image
- Branch B: CNN encoder on Fourier/Laplace filtered image
- Fusion layer merges both encoders
- Single decoder for segmentation

FIXED: Moved layer creation to __init__
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class FourierLaplaceLayer(layers.Layer):
    """Apply FFT and Laplacian filtering"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        # Convert to complex for FFT
        x_complex = tf.cast(x, tf.complex64)
        
        # 2D FFT
        x_freq = tf.signal.fft2d(x_complex[..., 0])
        
        # Shift zero frequency to center
        x_freq_shifted = tf.signal.fftshift(x_freq, axes=[1, 2])
        
        # Get magnitude (log scale for better visualization/learning)
        magnitude = tf.abs(x_freq_shifted)
        magnitude = tf.math.log1p(magnitude)
        
        # Normalize
        magnitude = magnitude / (tf.reduce_max(magnitude, axis=[1, 2], keepdims=True) + 1e-8)
        
        # Add channel dimension back
        magnitude = tf.expand_dims(magnitude, -1)
        
        return tf.cast(magnitude, tf.float32)


class LaplacianLayer(layers.Layer):
    """Apply Laplacian edge detection filter"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Laplacian kernel
        laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        self.kernel = tf.constant(laplacian_kernel.reshape(3, 3, 1, 1))
    
    def call(self, x):
        # Apply Laplacian filter
        laplacian = tf.nn.conv2d(x, self.kernel, strides=1, padding='SAME')
        # Normalize
        laplacian = laplacian / (tf.reduce_max(tf.abs(laplacian), axis=[1, 2, 3], keepdims=True) + 1e-8)
        return laplacian


class EncoderBlock(layers.Layer):
    """Standard encoder block"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.bn = layers.BatchNormalization()
        self.pool = layers.MaxPooling2D(2)
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x, training=training)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(layers.Layer):
    """Decoder block with skip connection"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.up = layers.UpSampling2D(2)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.bn = layers.BatchNormalization()
    
    def call(self, x, skip, training=False):
        x = self.up(x)
        x = layers.concatenate([x, skip])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x, training=training)
        return x


class DualEncoderFourier(Model):
    """Dual-Encoder: Spatial + Frequency domain features"""
    
    def __init__(self, input_shape=(512, 512, 1), n_filters=32, n_classes=1):
        super().__init__()
        
        self.n_filters = n_filters
        
        # Frequency preprocessing
        self.fourier = FourierLaplaceLayer()
        self.laplacian = LaplacianLayer()
        
        # Fix: Defined here instead of in call()
        self.freq_conv_in = layers.Conv2D(1, 1, padding='same')
        
        # Spatial Encoder (Branch A)
        self.enc_a1 = EncoderBlock(n_filters)
        self.enc_a2 = EncoderBlock(n_filters * 2)
        self.enc_a3 = EncoderBlock(n_filters * 4)
        self.enc_a4 = EncoderBlock(n_filters * 8)
        
        # Frequency Encoder (Branch B)
        self.enc_b1 = EncoderBlock(n_filters)
        self.enc_b2 = EncoderBlock(n_filters * 2)
        self.enc_b3 = EncoderBlock(n_filters * 4)
        self.enc_b4 = EncoderBlock(n_filters * 8)
        
        # Fusion bottleneck
        self.fusion_conv1 = layers.Conv2D(n_filters * 16, 3, padding='same', activation='relu')
        self.fusion_conv2 = layers.Conv2D(n_filters * 16, 3, padding='same', activation='relu')
        self.fusion_bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.3)
        
        # Skip connection fusions
        self.skip_fuse4 = layers.Conv2D(n_filters * 8, 1, padding='same')
        self.skip_fuse3 = layers.Conv2D(n_filters * 4, 1, padding='same')
        self.skip_fuse2 = layers.Conv2D(n_filters * 2, 1, padding='same')
        self.skip_fuse1 = layers.Conv2D(n_filters, 1, padding='same')
        
        # Decoder
        self.dec4 = DecoderBlock(n_filters * 8)
        self.dec3 = DecoderBlock(n_filters * 4)
        self.dec2 = DecoderBlock(n_filters * 2)
        self.dec1 = DecoderBlock(n_filters)
        
        # Output
        self.output_conv = layers.Conv2D(n_classes, 1, padding='same')
    
    def call(self, x, training=False):
        # Prepare frequency domain input
        x_freq = self.fourier(x)
        x_laplace = self.laplacian(x)
        x_combined_freq = layers.concatenate([x_freq, x_laplace])
        
        # Spatial Encoder
        a1, skip_a1 = self.enc_a1(x, training=training)
        a2, skip_a2 = self.enc_a2(a1, training=training)
        a3, skip_a3 = self.enc_a3(a2, training=training)
        a4, skip_a4 = self.enc_a4(a3, training=training)
        
        # Frequency Encoder
        x_freq_in = self.freq_conv_in(x_combined_freq)  # uses pre-defined layer
        b1, skip_b1 = self.enc_b1(x_freq_in, training=training)
        b2, skip_b2 = self.enc_b2(b1, training=training)
        b3, skip_b3 = self.enc_b3(b2, training=training)
        b4, skip_b4 = self.enc_b4(b3, training=training)
        
        # Fusion at bottleneck
        fused = layers.concatenate([a4, b4])
        fused = self.fusion_conv1(fused)
        fused = self.fusion_conv2(fused)
        fused = self.fusion_bn(fused, training=training)
        fused = self.dropout(fused, training=training)
        
        # Fuse skip connections
        skip4 = layers.concatenate([skip_a4, skip_b4])
        skip4 = self.skip_fuse4(skip4)
        
        skip3 = layers.concatenate([skip_a3, skip_b3])
        skip3 = self.skip_fuse3(skip3)
        
        skip2 = layers.concatenate([skip_a2, skip_b2])
        skip2 = self.skip_fuse2(skip2)
        
        skip1 = layers.concatenate([skip_a1, skip_b1])
        skip1 = self.skip_fuse1(skip1)
        
        # Decoder
        d4 = self.dec4(fused, skip4, training=training)
        d3 = self.dec3(d4, skip3, training=training)
        d2 = self.dec2(d3, skip2, training=training)
        d1 = self.dec1(d2, skip1, training=training)
        
        output = self.output_conv(d1)
        return output


def create_dual_encoder_fourier(input_shape=(512, 512, 1), n_filters=32, n_classes=1):
    """Factory function"""
    model = DualEncoderFourier(input_shape, n_filters, n_classes)
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    print(f"Dual-Encoder Fourier created: {model.count_params():,} parameters")
    return model


if __name__ == "__main__":
    model = create_dual_encoder_fourier(input_shape=(256, 256, 1))
    print(model.summary())
