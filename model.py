import tensorflow as tf
from config import Config

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(channels // 8, activation='relu')
        self.dense2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        
    def call(self, x):
        # Ensure consistent dtype throughout the layer
        dtype = x.dtype
        x = tf.cast(x, tf.float32)
        
        se = self.gap(x)
        se = self.dense1(se)
        se = self.dense2(se)
        attention = self.reshape(se)
        
        # Return result in original dtype
        return tf.cast(x * attention, dtype)

def build_model():
    # Set float32 policy for base model
    tf.keras.mixed_precision.set_global_policy('float32')
    
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)
    )
    
    # Unfreeze fewer layers to reduce overfitting
    for layer in base_model.layers[-45:]:
        layer.trainable = True
    
    inputs = tf.keras.Input(shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3))
    x = base_model(inputs)
    
    # Enhanced attention mechanism with proper layer
    x = AttentionBlock()(x)
    
    # Global pooling with additional regularization
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # First dense block with stronger regularization
    x = tf.keras.layers.Dense(1024, 
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA),
                            activity_regularizer=tf.keras.regularizers.l1(1e-5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Second dense block
    x = tf.keras.layers.Dense(512,
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.8)(x)
    
    # Additional layer for better feature discrimination
    x = tf.keras.layers.Dense(256,
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.6)(x)
    
    outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Restore mixed precision policy if needed
    if Config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    return model
