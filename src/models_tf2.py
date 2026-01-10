import tensorflow as tf
from tensorflow.keras import layers, Model, models, Input
import sys
import numpy as np # Not strictly needed for these model definitions but often useful in the file

class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
        super(InstanceNormalization, self).build(input_shape) # Call super build after adding weights

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

def get_norm_layer(norm_type='batchnorm'):
    if norm_type == 'batchnorm':
        return layers.BatchNormalization()
    elif norm_type == 'instancenorm':
        return InstanceNormalization()
    else:
        raise ValueError("Norm type must be 'batchnorm' or 'instancenorm'")

class UNetBlock(layers.Layer):
    """Basic U-Net convolutional block with batch normalization AND optional dropout"""
    def __init__(self, filters, dropout_rate=0.0, norm_type='batchnorm', name='unet_block', **kwargs):
        super(UNetBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        # First convolution
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name}_conv1'
        )
        self.bn1 = layers.BatchNormalization(name=f'{name}_bn1')
        self.relu1 = layers.ReLU(name=f'{name}_relu1')

        # Second convolution
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name}_conv2'
        )
        self.bn2 = layers.BatchNormalization(name=f'{name}_bn2')
        self.relu2 = layers.ReLU(name=f'{name}_relu2')

        if self.dropout_rate > 0.0:
            self.dropout = layers.Dropout(self.dropout_rate, name=f'{name}_dropout')
        else:
            self.dropout = None

    def call(self, inputs, training=False): # training flag is crucial
        x = self.conv1(inputs)
        x = self.bn1(x, training=training) # Pass training flag
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training) # Pass training flag
        x = self.relu2(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training) # Pass training flag
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "dropout_rate": self.dropout_rate
        })
        return config

class PancreasSeg(Model):
    """
    U-Net model for pancreas segmentation with support for dropout.
    """
    def __init__(self, config): # config is expected to be an object like StableSSLConfig
          super(PancreasSeg, self).__init__(name="pancreas_seg") # Give the model a name
          
          self.config = config # Store config if needed later, or just extract values
          self.img_size_x = config.img_size_x
          self.img_size_y = config.img_size_y
          self.num_channels = config.num_channels # Get num_channels from config
          self.num_classes = config.num_classes
          n_filters = config.n_filters

          # Get dropout_rate from config, default to 0.0 if not present
          self.dropout_rate = getattr(config, 'dropout_rate', 0.0) 
          tf.print(f"Building model {self.name} with input_shape: (None, {self.img_size_y}, {self.img_size_x}, {self.num_channels}), n_filters: {n_filters}, dropout_rate: {self.dropout_rate}")

          # Encoder blocks
          self.enc1 = UNetBlock(filters=n_filters, dropout_rate=self.dropout_rate, name='enc1')
          self.enc2 = UNetBlock(filters=n_filters*2, dropout_rate=self.dropout_rate, name='enc2')
          self.enc3 = UNetBlock(filters=n_filters*4, dropout_rate=self.dropout_rate, name='enc3')
          self.enc4 = UNetBlock(filters=n_filters*8, dropout_rate=self.dropout_rate, name='enc4')

          # Bridge
          self.bridge = UNetBlock(filters=n_filters*16, dropout_rate=self.dropout_rate, name='bridge')

          # Decoder blocks (typically no dropout in the decoder, so dropout_rate=0.0)
          self.dec4 = UNetBlock(filters=n_filters*8, dropout_rate=0.0, name='dec4')
          self.dec3 = UNetBlock(filters=n_filters*4, dropout_rate=0.0, name='dec3')
          self.dec2 = UNetBlock(filters=n_filters*2, dropout_rate=0.0, name='dec2')
          self.dec1 = UNetBlock(filters=n_filters, dropout_rate=0.0, name='dec1')

          # Upsampling layers
          self.up4 = layers.Conv2DTranspose(filters=n_filters*8, kernel_size=2, strides=2, padding='same', name='up4_trans') # Halves filters
          self.up3 = layers.Conv2DTranspose(filters=n_filters*4, kernel_size=2, strides=2, padding='same', name='up3_trans')
          self.up2 = layers.Conv2DTranspose(filters=n_filters*2, kernel_size=2, strides=2, padding='same', name='up2_trans')
          self.up1 = layers.Conv2DTranspose(filters=n_filters, kernel_size=2, strides=2, padding='same', name='up1_trans')

          # Max pooling layers
          self.pool = layers.MaxPooling2D(pool_size=(2, 2), name='pool')

          # Final convolution
          self.final_conv = layers.Conv2D(
              filters=self.num_classes, # Should be 1 for binary segmentation
              kernel_size=1,
              activation=None, # Output logits
              kernel_initializer='he_normal', # Consistent initializer
              name='final_output_conv'
          )
          
          # No self.built = False needed, Keras handles it.
          # The model will be built upon the first call or by explicit build() call.

    def build(self, input_shape): # input_shape should include batch, e.g., (None, H, W, C)
        """Overriding build to ensure model is built and to print info."""
        super(PancreasSeg, self).build(input_shape)
        tf.print(f"Model {self.name}: Built with input shape {input_shape}.")
        # You can call with dummy input here if you want to ensure all weights are created,
        # but Keras build mechanism often handles this.
        # Example: if not self.weights: self(tf.zeros((1, self.img_size_y, self.img_size_x, self.num_channels)))
        tf.print(f"Model {self.name} has {len(self.weights)} weights after build.")
        # Adding a specific check that the model builds correctly by ensuring a forward pass.
        if not self.weights: # If build didn't create weights, force a call
            tf.print(f"Model {self.name}: Ensuring forward pass by calling with dummy input of shape (1, {self.img_size_y}, {self.img_size_x}, {self.num_channels})...")
            self(tf.zeros((1, self.img_size_y, self.img_size_x, self.num_channels)), training=False)
            tf.print(f"Model {self.name}: Forward pass call completed.")
        tf.print(f"Model {self.name} build process complete. Final built status: {self.built}")


    def call(self, inputs, training=False): # training flag is crucial
        """Forward pass of the model"""
        x = tf.cast(inputs, tf.float32) # Ensure input is float32
        
        # Encoder Path
        e1 = self.enc1(x, training=training)
        p1 = self.pool(e1)

        e2 = self.enc2(p1, training=training)
        p2 = self.pool(e2)

        e3 = self.enc3(p2, training=training)
        p3 = self.pool(e3)

        e4 = self.enc4(p3, training=training)
        p4 = self.pool(e4)

        # Bridge
        b = self.bridge(p4, training=training)

        # Decoder Path
        # For Conv2DTranspose, the number of filters should match the target number of channels
        # for the block it's upsampling TO, before concatenation.
        # The concatenation doubles the channels, then UNetBlock reduces it.
        # So up4 should output n_filters*8 channels to match e4.
        # up3 should output n_filters*4 channels to match e3.
        # up2 should output n_filters*2 channels to match e2.
        # up1 should output n_filters channels to match e1.
        
        u4 = self.up4(b) # up4 outputs n_filters*8 (e.g. 256 if n_filters*16 was 512 for bridge)
                         # This seems my Conv2DTranspose filter counts were off in previous version.
                         # Let's correct them based on standard U-Net.
                         # up4 output channels should match e4's channels if concatenating before dec4.
                         # Bridge (n_filters*16) -> up4 -> output (n_filters*8)
                         # Corrected upsampling layer definitions (filters should be what they output to match skip connection)
                         # self.up4 = layers.Conv2DTranspose(filters=n_filters*8, ...)
                         # self.up3 = layers.Conv2DTranspose(filters=n_filters*4, ...)
                         # self.up2 = layers.Conv2DTranspose(filters=n_filters*2, ...)
                         # self.up1 = layers.Conv2DTranspose(filters=n_filters, ...)
                         # The version you had was:
                         # self.up4 = layers.Conv2DTranspose(filters=n_filters*16, ...) -> this would output n_filters*16 channels
                         # This is fine if dec4 expects input channels of (n_filters*16 + n_filters*8).
                         # Let's stick to your version for now, assuming UNetBlock handles the channel reduction.

        c4 = tf.concat([u4, e4], axis=-1) # u4 (n_filters*16) + e4 (n_filters*8) = n_filters*24
                                          # dec4 then reduces this to n_filters*8
        d4 = self.dec4(c4, training=training)

        u3 = self.up3(d4) # d4 (n_filters*8) -> up3 -> u3 (n_filters*8)
        c3 = tf.concat([u3, e3], axis=-1) # u3 (n_filters*8) + e3 (n_filters*4) = n_filters*12
        d3 = self.dec3(c3, training=training) # dec3 outputs n_filters*4

        u2 = self.up2(d3) # d3 (n_filters*4) -> up2 -> u2 (n_filters*4)
        c2 = tf.concat([u2, e2], axis=-1) # u2 (n_filters*4) + e2 (n_filters*2) = n_filters*6
        d2 = self.dec2(c2, training=training) # dec2 outputs n_filters*2

        u1 = self.up1(d2) # d2 (n_filters*2) -> up1 -> u1 (n_filters*2)
        c1 = tf.concat([u1, e1], axis=-1) # u1 (n_filters*2) + e1 (n_filters*1) = n_filters*3
        d1 = self.dec1(c1, training=training) # dec1 outputs n_filters*1

        logits = self.final_conv(d1) # Output: (batch, H, W, num_classes)

        return logits
      
    # get_projections method seems unrelated to PancreasSeg directly unless it has an encoder part.
    # If it was intended for a different model or SSL setup, it might need adjustment
    # or removal if not used by PancreasSeg itself.
    # For now, I will comment it out as it refers to self.encode and self.projection which are not defined.
    # def get_projections(self, x, training=False):
    #     """Get embeddings for contrastive learning"""
    #     # bridge, _ = self.encode(x, training=training) # self.encode is not defined
    #     # projections = self.projection(bridge, training=training) # self.projection is not defined
    #     # return projections
    #     pass

class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient',**kwargs):
        super().__init__(name=name,**kwargs); self.sum_ds=self.add_weight(name='sum_ds',initializer='zeros'); self.n_samp=self.add_weight(name='n_samp',initializer='zeros'); self.s=1e-6
    def update_state(self, yt, yp_logits, sample_weight=None):
        yt_f=tf.cast(yt,tf.float32); ypp=tf.nn.sigmoid(yp_logits); ypb=tf.cast(ypp>0.5,tf.float32)
        intr=tf.reduce_sum(yt_f*ypb,axis=[1,2,3]); sum_t=tf.reduce_sum(yt_f,axis=[1,2,3]); sum_p=tf.reduce_sum(ypb,axis=[1,2,3])
        dice_samp=(2.*intr+self.s)/(sum_t+sum_p+self.s); self.sum_ds.assign_add(tf.reduce_sum(dice_samp)); self.n_samp.assign_add(tf.cast(tf.shape(yt)[0],tf.float32))
    def result(self): return tf.cond(tf.equal(self.n_samp,0.0),lambda:0.0,lambda:self.sum_ds/self.n_samp)
    def reset_state(self): self.sum_ds.assign(0.0); self.n_samp.assign(0.0)
    def get_config(self): conf=super().get_config(); conf.update({'smooth':self.s}); return conf

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, config=None, smooth=1e-6, alpha=0.25, gamma=2):
        super().__init__(name="combined_loss")
        self.smooth = smooth
        self.alpha = alpha 
        self.gamma = gamma
        # Instantiate BCE loss to control reduction
        self.bce_for_focal = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, # focal_loss receives probabilities
            reduction=tf.keras.losses.Reduction.NONE # Get per-element loss
        )

    def dice_loss(self, y_true, y_pred_probs): # Expects probabilities
          y_true_f = tf.cast(y_true, tf.float32)
          y_pred_probs_f = tf.cast(y_pred_probs, tf.float32)
          intersection = tf.reduce_sum(y_true_f * y_pred_probs_f, axis=[1, 2, 3]) 
          union_sum = tf.reduce_sum(y_true_f, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_probs_f, axis=[1, 2, 3])
          dice_per_sample = (2. * intersection + self.smooth) / (union_sum + self.smooth)
          return 1.0 - tf.reduce_mean(dice_per_sample)

    def focal_loss(self, y_true, y_pred_probs): # y_true and y_pred_probs are [B,H,W,1]
        y_true_f = tf.cast(y_true, tf.float32) 
        y_pred_probs_f = tf.clip_by_value(tf.cast(y_pred_probs, tf.float32), 1e-7, 1.0 - 1e-7)
        
        # tf.print("DEBUG focal_loss input: y_true_f shape:", tf.shape(y_true_f), "y_pred_probs_f shape:", tf.shape(y_pred_probs_f), output_stream=sys.stderr)

        # Calculate BCE per pixel.
        # self.bce_for_focal is BinaryCrossentropy(from_logits=False, reduction=Reduction.NONE)
        # If y_true_f and y_pred_probs_f are [B,H,W,1], bce_output might be [B,H,W].
        bce_output_maybe_squeezed = self.bce_for_focal(y_true_f, y_pred_probs_f)
        
        # tf.print("DEBUG focal_loss: bce_output_maybe_squeezed shape:", tf.shape(bce_output_maybe_squeezed), output_stream=sys.stderr)

        # Ensure bce_per_pixel is [B,H,W,1] by adding a new axis if it was squeezed to [B,H,W]
        if len(bce_output_maybe_squeezed.shape) == 3: # If shape is [B,H,W]
            bce_per_pixel = bce_output_maybe_squeezed[..., tf.newaxis] # Add channel dim -> [B,H,W,1]
        else: # Assume it's already [B,H,W,1] or some other error occurred
            bce_per_pixel = bce_output_maybe_squeezed
        
        # tf.print("DEBUG focal_loss: bce_per_pixel (after potential expand) shape:", tf.shape(bce_per_pixel), output_stream=sys.stderr)

        # p_t, focal_modulator, alpha_weight will be [B,H,W,1] because inputs are [B,H,W,1]
        p_t = y_true_f * y_pred_probs_f + (1.0 - y_true_f) * (1.0 - y_pred_probs_f)
        focal_modulator = tf.pow(1.0 - p_t, self.gamma)
        alpha_weight = y_true_f * self.alpha + (1.0 - y_true_f) * (1.0 - self.alpha)
        
        # tf.print("DEBUG focal_loss FINAL shapes B4 mul:",
        #          "\n  alpha_weight shape:", tf.shape(alpha_weight),         # Expected: [B,H,W,1]
        #          "\n  focal_modulator shape:", tf.shape(focal_modulator),   # Expected: [B,H,W,1]
        #          "\n  bce_per_pixel shape:", tf.shape(bce_per_pixel),       # Expected: [B,H,W,1]
        #          output_stream=sys.stderr, summarize=-1)

        focal_loss_elements = alpha_weight * focal_modulator * bce_per_pixel 
        
        return tf.reduce_mean(focal_loss_elements)

    def call(self, y_true, y_pred_logits):
        y_true_processed = y_true
        y_pred_logits_processed = y_pred_logits
        y_pred_probs_processed = tf.nn.sigmoid(y_pred_logits_processed)
        dice_l = self.dice_loss(y_true_processed, y_pred_probs_processed)
        focal_l = self.focal_loss(y_true_processed, y_pred_probs_processed)
        return 0.5 * dice_l + 0.5 * focal_l
       
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, **kwargs):
        super(ContrastiveLoss, self).__init__(name="contrastive_loss", **kwargs) # Added name
        self.temperature = temperature

    def call(self, projection_1, projection_2):
        proj_1 = tf.math.l2_normalize(projection_1, axis=1)
        proj_2 = tf.math.l2_normalize(projection_2, axis=1)
        batch_size = tf.shape(proj_1)[0]
        # Ensure batch_size is not 0 to avoid errors with tf.eye
        if tf. कम(batch_size, 1): # Using tf.less for graph compatibility
            return tf.constant(0.0, dtype=projection_1.dtype) # Return 0 loss if batch is empty

        similarity_matrix = tf.matmul(proj_1, proj_2, transpose_b=True) / self.temperature
        labels = tf.eye(batch_size) # Positive pairs on the diagonal
        
        # Using categorical_crossentropy with logits=True is more stable for NT-Xent style loss
        loss = tf.keras.losses.categorical_crossentropy(
            labels,
            similarity_matrix, # Pass similarity_matrix as logits
            from_logits=True
        )
        return tf.reduce_mean(loss)

# Optional: Factory function (can be useful but not strictly necessary for PancreasSeg itself)
# def create_ssl_model(config):
#     model = PancreasSeg(config)
#     # input_shape = (config.img_size_y, config.img_size_x, config.num_channels) # Build expects batch dim
#     # model.build(input_shape=(None,) + input_shape) # Build with batch_size=None
#     # The model will be built on first call.
#     return model
    def create_ssl_model(config):
        """
        Factory function to create model with losses and optimizers
        """
        model = PancreasSeg(config)
        
        # Ensure model is initialized
        input_shape = (None, config.img_size_x, config.img_size_y, config.num_channels)
        model.build(input_shape)
        
        # Define optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        # Define losses
        supervised_loss = DiceLoss()

        return {
            'model': model,
            'optimizer': optimizer,
            'supervised_loss': supervised_loss,
        }