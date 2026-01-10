import os
import tensorflow as tf
from unetr import create_unetr

# Force eager to start
# tf.config.run_functions_eagerly(True)

def test():
    print("Creating model...")
    model = create_unetr((256, 256, 1))
    
    print("Creating dummy data...")
    x = tf.random.normal((2, 256, 256, 1))
    y = tf.random.uniform((2, 256, 256, 1), minval=0, maxval=2, dtype=tf.int32)
    y = tf.cast(y, tf.float32)
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    print("Running EAGER train step...")
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Eager step done. Loss: {loss.numpy()}")
    
    print("Running GRAPH train step (tf.function)...")
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = loss_fn(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    l = train_step(x, y)
    print(f"Graph step 1 done. Loss: {l}")
    l = train_step(x, y)
    print(f"Graph step 2 done. Loss: {l}")
    print("TEST PASSED")

if __name__ == "__main__":
    test()
