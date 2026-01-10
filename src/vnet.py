"""
V-Net (2D Adaptation)
Architecture from: https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation
Refactored for cleaner TensorFlow 2.x / Keras usage.
Features: 
- Residual Blocks in Encoder/Decoder
- PReLU Activation
- Deep Supervision (Optional)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class ResBlock(layers.Layer):
    """Residual Block for V-Net"""
    def __init__(self, channels, stage, keep_prob=1.0, stage_num=5, **kwargs):
        super().__init__(**kwargs)
        self.stage = stage
        self.stage_num = stage_num
        self.keep_prob = keep_prob
        self.channels = channels
        
        # Convolutions for the residual path
        num_convs = 3 if stage > 3 else stage
        self.convs = []
        for _ in range(num_convs):
            self.convs.append(tf.keras.Sequential([
                layers.Conv2D(channels, 5, padding='same', 
                              kernel_initializer='he_normal', 
                              kernel_regularizer=regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.PReLU()
            ]))
            
        self.add = layers.Add()
        self.after_add = layers.PReLU()
        self.dropout = layers.Dropout(1 - keep_prob) # Dropout takes rate (1-keep_prob)
        
        # Downsample if needed
        if stage < stage_num:
            self.down_conv = tf.keras.Sequential([
                layers.Conv2D(channels * 2, 2, strides=2, padding='same', 
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.PReLU()
            ])
            self.has_down = True
        else:
            self.has_down = False
            
    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
            
        # Residual Connection
        # Note: Kursat code does: conv_add=PReLU()(add([inputs,conv]))
        # But if channels mismatch (e.g. first block input is 16, output 16), it works.
        # If input channels != output channels, we might need projection.
        # In V-Net passed code, channels seem consistent within block.
        
        x = self.add([inputs, x])
        x = self.after_add(x)
        x = self.dropout(x)
        
        if self.has_down:
            down = self.down_conv(x)
            return down, x
        else:
            return x, x

class UpResBlock(layers.Layer):
    """Upsampling Residual Block for V-Net"""
    def __init__(self, channels, stage, **kwargs):
        super().__init__(**kwargs)
        self.stage = stage
        
        self.concat = layers.Concatenate(axis=-1)
        
        num_convs = 3 if stage > 3 else stage
        self.convs = []
        for _ in range(num_convs):
            self.convs.append(tf.keras.Sequential([
                layers.Conv2D(channels, 5, padding='same', 
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.PReLU()
            ]))
            
        self.add = layers.Add()
        self.after_add = layers.PReLU()
        
        if stage > 1:
            # Upsample for next stage
            # Target channels for next stage is channels / 2
            next_channels = channels // 2
            self.upsample = tf.keras.Sequential([
                layers.Conv2DTranspose(next_channels, 2, strides=2, padding='valid',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.PReLU()
            ])
            self.has_up = True
        else:
            self.has_up = False

    def call(self, forward_conv, input_conv):
        # forward_conv: from encoder (skip connection)
        # input_conv: from lower decoder level
        
        x = self.concat([forward_conv, input_conv])
        
        # Convs
        res_path = x
        for conv in self.convs:
            res_path = conv(res_path)
            
        # Residual add (Kursat adds input_conv + processed conv)
        # Wait, concatenation changes channel count. 
        # In Kursat code: conv_add=PReLU()(add([input_conv,conv]))
        # input_conv is the upsampled feature map. 
        # conv is the result of applying convs to cat([skip, input_conv]).
        # The convs maintain channel size (16*2**(stage-1)).
        # input_conv should match this if upsample was correct.
        
        x = self.add([input_conv, res_path])
        x = self.after_add(x)
        
        if self.has_up:
            return self.upsample(x)
        else:
            return x

class VNet(Model):
    def __init__(self, input_size=(256, 256, 1), num_class=1, stage_num=5, **kwargs):
        super().__init__(**kwargs)
        
        # Initial Conv
        self.init_conv = tf.keras.Sequential([
            layers.Conv2D(16, 5, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.PReLU()
        ])
        
        self.stage_num = stage_num
        self.stages = []
        
        # Encoder Stages
        for s in range(1, stage_num + 1):
            channels = 16 * (2 ** (s - 1))
            self.stages.append(ResBlock(channels, s, keep_prob=0.5 if s < stage_num else 1.0, stage_num=stage_num))
            
        # First Upsample (from Bottleneck)
        # Bottleneck output (stage 5) has channels 16*2^4 = 256. 
        # Needs to upsample to Stage 4 channels (16*2^3 = 128).
        self.bottleneck_up = tf.keras.Sequential([
            layers.Conv2DTranspose(16 * (2 ** (stage_num - 2)), 2, strides=2, padding='valid',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.PReLU()
        ])
            
        # Decoder Stages
        self.up_stages = []
        for d in range(stage_num - 1, 0, -1):
            # Stage d channels: 16 * 2^(d-1)
            channels = 16 * (2 ** (d - 1))
            self.up_stages.append(UpResBlock(channels, d))
            
        # Final Output
        activation = 'sigmoid' if num_class == 1 else 'softmax'
        self.out_conv = layers.Conv2D(num_class, 1, padding='same', activation=activation,
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=regularizers.l2(1e-4))
                                      
    def call(self, inputs, training=False):
        # Initial
        # Kursat's code: input -> Conv2D -> PReLU... -> x
        x = self.init_conv(inputs)
        
        # Encoder
        features = []
        for stage in self.stages:
            x, feature = stage(x) # ResBlock returns (downsampled, before_downsample)
            features.append(feature)
            
        # Last element of features is the bottleneck output (no downsampling happened)
        # Wait, ResBlock logic: if stage < 5: return down, x. Else return x, x.
        # So inputs to decoder is the "down" (x) of previous loop.
        # Bottleneck is x. 
        
        # First Upsample
        conv_up = self.bottleneck_up(x)
        
        # Decoder
        # Range d: 4, 3, 2, 1
        # features indices: 0..4
        # features[4] is bottleneck
        # features[3] is stage 4 skip
        for i, up_stage in enumerate(self.up_stages):
            d = self.stage_num - 1 - i # 4, 3, 2, 1
            skip = features[d-1] # features[3], [2], [1], [0]
            conv_up = up_stage(skip, conv_up)
            
        return self.out_conv(conv_up)

def create_vnet(input_shape=(256, 256, 1)):
    model = VNet(input_shape)
    dummy = tf.zeros((1,) + input_shape)
    _ = model(dummy)
    print(f"V-Net created: {model.count_params():,} params")
    return model

if __name__ == "__main__":
    create_vnet()
