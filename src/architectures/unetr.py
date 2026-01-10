"""
UNETR: Transformers for 3D Medical Image Segmentation (Adapted for 2D)
Refs:
- https://arxiv.org/abs/2103.10504
- Uses separate Transformer encoder + CNN decoder
FIXED: All layers defined in __init__ to support tf.function
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

class PatchEmbedding(layers.Layer):
    """Split image into patches and embed them"""
    def __init__(self, patch_size=16, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.proj = layers.Dense(hidden_size)

    def call(self, x):
        # x shape: (B, H, W, C)
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        B = tf.shape(patches)[0]
        patches = tf.reshape(patches, [B, -1, patches.shape[-1]])
        embeddings = self.proj(patches)
        return embeddings

class TransformerBlock(layers.Layer):
    """Standard Transformer Encoder Block"""
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(hidden_size),
            layers.Dropout(dropout)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x_norm = self.norm1(x)
        attn_out = self.att(x_norm, x_norm, training=training)
        x = x + self.dropout1(attn_out, training=training)
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm, training=training)
        x = x + mlp_out
        return x

class UNETR(Model):
    """UNETR adapted for 2D segmentation"""
    def __init__(self, input_shape=(256, 256, 1), patch_size=16, hidden_size=768, num_heads=12, mlp_dim=3072, num_layers=12, n_classes=1):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_shape_ = input_shape
        
        # Encoder (Transformer)
        self.patch_embed = PatchEmbedding(patch_size, hidden_size)
        
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(1, num_patches, hidden_size), initializer="zeros"
        )
        
        self.blocks = [
            TransformerBlock(hidden_size, num_heads, mlp_dim)
            for _ in range(num_layers)
        ]
        
        # Extract indices: 3, 6, 9, 12 (0-based: 2, 5, 8, 11)
        self.extract_indices = [2, 5, 8, 11]
        
        # Decoder Layers - ALL defined here
        n_filters = 32
        
        # 1. Bottleneck Upsample (from z12)
        # z12 is 16x16 (if 256 input / 16 patch)
        self.bottleneck_up = layers.Conv2DTranspose(512, 2, strides=2, padding='same', activation='relu') # 16->32
        
        # 2. Level 3 (target 32x32)
        self.z9_up1 = layers.Conv2DTranspose(256, 2, strides=2, padding='same') # z9 (16) -> 32
        # Concat z9_up1 + bottleneck_up -> (32,32, 256+512)
        self.dec3_conv = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.up_to_64 = layers.UpSampling2D(2) # 32->64
        
        # 3. Level 2 (target 64x64)
        self.z6_up1 = layers.Conv2DTranspose(128, 2, strides=2, padding='same') # z6 (16) -> 32
        self.z6_up2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same') # 32 -> 64
        # Concat z6_up2 + up_to_64 -> (64,64, 128+256)
        self.dec2_conv = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.up_to_128 = layers.UpSampling2D(2) # 64->128
        
        # 4. Level 1 (target 128x128)
        self.z3_up1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same') # z3 (16) -> 32
        self.z3_up2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same') # 32 -> 64
        self.z3_up3 = layers.Conv2DTranspose(64, 2, strides=2, padding='same') # 64 -> 128
        # Concat z3_up3 + up_to_128 -> (128,128, 64+128)
        self.dec1_conv = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.up_to_256 = layers.UpSampling2D(2) # 128->256
        
        # 5. Level 0 (target 256x256)
        # Using simple convs for final refinement
        self.dec0_conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        
        # Final Output
        self.out_conv = layers.Conv2D(n_classes, 1, padding='same')

    def reshape_transform(self, x, H, W):
        # x: (B, N, C) -> (B, H, W, C)
        B = tf.shape(x)[0]
        return tf.reshape(x, [B, H, W, self.hidden_size])

    def call(self, x, training=False):
        # 1. Embed
        x_emb = self.patch_embed(x)
        x_emb = x_emb + self.pos_embed
        
        # 2. Encoder
        skips = []
        hidden_states = x_emb
        
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, training=training)
            if i in self.extract_indices:
                skips.append(hidden_states)
        
        # Reshape to spatial
        grid_h = self.input_shape_[0] // self.patch_size
        grid_w = self.input_shape_[1] // self.patch_size
        
        # All skips are 16x16 spatial
        z3 = self.reshape_transform(skips[0], grid_h, grid_w) 
        z6 = self.reshape_transform(skips[1], grid_h, grid_w)
        z9 = self.reshape_transform(skips[2], grid_h, grid_w)
        z12 = self.reshape_transform(skips[3], grid_h, grid_w)
        
        # Decoder 
        # -- Level 3 (32x32) --
        # Upsample bottleneck (z12)
        x = self.bottleneck_up(z12) # 16->32
        # Projection for skip (z9)
        skip3 = self.z9_up1(z9) # 16->32
        x = layers.concatenate([x, skip3])
        x = self.dec3_conv(x)
        
        # -- Level 2 (64x64) --
        x = self.up_to_64(x) # 32->64
        # Projection for skip (z6)
        skip2 = self.z6_up1(z6)
        skip2 = self.z6_up2(skip2) # 32->64
        x = layers.concatenate([x, skip2])
        x = self.dec2_conv(x)
        
        # -- Level 1 (128x128) --
        x = self.up_to_128(x) # 64->128
        # Projection for skip (z3)
        skip1 = self.z3_up1(z3)
        skip1 = self.z3_up2(skip1)
        skip1 = self.z3_up3(skip1) # 64->128
        x = layers.concatenate([x, skip1])
        x = self.dec1_conv(x)
        
        # -- Level 0 (256x256) --
        x = self.up_to_256(x) # 128->256
        x = self.dec0_conv1(x)
        
        # Output
        return self.out_conv(x)


def create_unetr(input_shape=(256, 256, 1)):
    model = UNETR(input_shape)
    dummy = tf.zeros((1,) + input_shape)
    _ = model(dummy)
    print(f"UNETR created: {model.count_params():,} params")
    return model

if __name__ == "__main__":
    create_unetr()
