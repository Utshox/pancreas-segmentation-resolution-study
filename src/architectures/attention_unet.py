"""
Attention U-Net for Pancreas Segmentation
Adds attention gates to skip connections to focus on relevant features
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


class AttentionGate(layers.Layer):
    """Attention Gate for focusing on relevant features"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        self.W_g = layers.Conv2D(self.filters, 1, padding='same')
        self.W_x = layers.Conv2D(self.filters, 1, padding='same')
        self.psi = layers.Conv2D(1, 1, padding='same')
        self.relu = layers.ReLU()
        self.sigmoid = layers.Activation('sigmoid')
        self.bn = layers.BatchNormalization()
        
    def call(self, x, g, training=False):
        # x: skip connection features
        # g: gating signal from decoder
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g if needed
        if g1.shape[1] != x1.shape[1]:
            g1 = tf.image.resize(g1, [x1.shape[1], x1.shape[2]])
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        psi = self.bn(psi, training=training)
        
        return x * psi


class AttentionUNet(Model):
    """Attention U-Net with attention gates at skip connections"""
    
    def __init__(self, input_shape=(512, 512, 1), n_filters=32, n_classes=1, dropout_rate=0.3):
        super().__init__()
        
        self.input_shape_ = input_shape
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.enc1_conv1 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')
        self.enc1_conv2 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')
        self.enc1_bn = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        
        self.enc2_conv1 = layers.Conv2D(n_filters*2, 3, padding='same', activation='relu')
        self.enc2_conv2 = layers.Conv2D(n_filters*2, 3, padding='same', activation='relu')
        self.enc2_bn = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        
        self.enc3_conv1 = layers.Conv2D(n_filters*4, 3, padding='same', activation='relu')
        self.enc3_conv2 = layers.Conv2D(n_filters*4, 3, padding='same', activation='relu')
        self.enc3_bn = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        
        self.enc4_conv1 = layers.Conv2D(n_filters*8, 3, padding='same', activation='relu')
        self.enc4_conv2 = layers.Conv2D(n_filters*8, 3, padding='same', activation='relu')
        self.enc4_bn = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(2)
        
        # Bottleneck
        self.bottleneck_conv1 = layers.Conv2D(n_filters*16, 3, padding='same', activation='relu')
        self.bottleneck_conv2 = layers.Conv2D(n_filters*16, 3, padding='same', activation='relu')
        self.bottleneck_bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        
        # Attention Gates
        self.att4 = AttentionGate(n_filters*8)
        self.att3 = AttentionGate(n_filters*4)
        self.att2 = AttentionGate(n_filters*2)
        self.att1 = AttentionGate(n_filters)
        
        # Decoder
        self.up4 = layers.UpSampling2D(2)
        self.dec4_conv1 = layers.Conv2D(n_filters*8, 3, padding='same', activation='relu')
        self.dec4_conv2 = layers.Conv2D(n_filters*8, 3, padding='same', activation='relu')
        self.dec4_bn = layers.BatchNormalization()
        
        self.up3 = layers.UpSampling2D(2)
        self.dec3_conv1 = layers.Conv2D(n_filters*4, 3, padding='same', activation='relu')
        self.dec3_conv2 = layers.Conv2D(n_filters*4, 3, padding='same', activation='relu')
        self.dec3_bn = layers.BatchNormalization()
        
        self.up2 = layers.UpSampling2D(2)
        self.dec2_conv1 = layers.Conv2D(n_filters*2, 3, padding='same', activation='relu')
        self.dec2_conv2 = layers.Conv2D(n_filters*2, 3, padding='same', activation='relu')
        self.dec2_bn = layers.BatchNormalization()
        
        self.up1 = layers.UpSampling2D(2)
        self.dec1_conv1 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')
        self.dec1_conv2 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')
        self.dec1_bn = layers.BatchNormalization()
        
        # Output
        self.output_conv = layers.Conv2D(n_classes, 1, padding='same')
        
    def call(self, x, training=False):
        # Encoder
        e1 = self.enc1_conv1(x)
        e1 = self.enc1_conv2(e1)
        e1 = self.enc1_bn(e1, training=training)
        p1 = self.pool1(e1)
        
        e2 = self.enc2_conv1(p1)
        e2 = self.enc2_conv2(e2)
        e2 = self.enc2_bn(e2, training=training)
        p2 = self.pool2(e2)
        
        e3 = self.enc3_conv1(p2)
        e3 = self.enc3_conv2(e3)
        e3 = self.enc3_bn(e3, training=training)
        p3 = self.pool3(e3)
        
        e4 = self.enc4_conv1(p3)
        e4 = self.enc4_conv2(e4)
        e4 = self.enc4_bn(e4, training=training)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck_conv1(p4)
        b = self.bottleneck_conv2(b)
        b = self.bottleneck_bn(b, training=training)
        b = self.dropout(b, training=training)
        
        # Decoder with Attention
        d4 = self.up4(b)
        e4_att = self.att4(e4, d4, training=training)
        d4 = layers.concatenate([d4, e4_att])
        d4 = self.dec4_conv1(d4)
        d4 = self.dec4_conv2(d4)
        d4 = self.dec4_bn(d4, training=training)
        
        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3, training=training)
        d3 = layers.concatenate([d3, e3_att])
        d3 = self.dec3_conv1(d3)
        d3 = self.dec3_conv2(d3)
        d3 = self.dec3_bn(d3, training=training)
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2, training=training)
        d2 = layers.concatenate([d2, e2_att])
        d2 = self.dec2_conv1(d2)
        d2 = self.dec2_conv2(d2)
        d2 = self.dec2_bn(d2, training=training)
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1, training=training)
        d1 = layers.concatenate([d1, e1_att])
        d1 = self.dec1_conv1(d1)
        d1 = self.dec1_conv2(d1)
        d1 = self.dec1_bn(d1, training=training)
        
        output = self.output_conv(d1)
        return output


def create_attention_unet(input_shape=(512, 512, 1), n_filters=32, n_classes=1):
    """Factory function to create Attention U-Net"""
    model = AttentionUNet(input_shape, n_filters, n_classes)
    # Build model
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    print(f"Attention U-Net created: {model.count_params():,} parameters")
    return model


if __name__ == "__main__":
    # Test
    model = create_attention_unet(input_shape=(256, 256, 1))
    print(model.summary())
