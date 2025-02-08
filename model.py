import tensorflow as tf
from config import Config


'''
class AttentionBlock(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation Attention mechanism for feature refinement.
    Helps the model focus on informative features and suppress less useful ones.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        # Global average pooling for channel-wise statistics
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        # Two-layer MLP for computing attention weights
        self.dense1 = tf.keras.layers.Dense(channels // 8, activation='relu')  
        self.dense2 = tf.keras.layers.Dense(channels, activation='sigmoid')  
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))  
        
    def call(self, x):
        # Ensure consistent dtype throughout the layer
        dtype = x.dtype
        x = tf.cast(x, tf.float32)
        
        # Compute channel-wise attention weights
        se = self.gap(x)
        se = self.dense1(se)
        se = self.dense2(se)
        attention = self.reshape(se)
        
        # Apply attention weights to input features
        return tf.cast(x * attention, dtype)

def build_model():
    """
    Builds a CNN model for brain tumor classification using transfer learning.
    Architecture:
    1. ResNet50V2 base model (pre-trained on ImageNet)
    2. Custom attention mechanism
    3. Multiple dense layers with regularization
    4. Softmax output for 4-class classification
    """
    # Set float32 policy for base model
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Initialize pre-trained ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,  
        weights='imagenet',  
        input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)
    )
    
    # Fine-tune the last 45 layers while keeping others frozen
    for layer in base_model.layers[-45:]:
        layer.trainable = True
    
    # Build model architecture
    inputs = tf.keras.Input(shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3))
    x = base_model(inputs)
    
    # Apply attention mechanism to focus on relevant features
    x = AttentionBlock()(x)
    
    # Global pooling to reduce spatial dimensions
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # First dense block with strong regularization
    x = tf.keras.layers.Dense(1024, 
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA),
                            activity_regularizer=tf.keras.regularizers.l1(1e-5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    # Second dense block with moderate regularization
    x = tf.keras.layers.Dense(512,
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.8)(x)  
    
    # Final dense block for feature refinement
    x = tf.keras.layers.Dense(256,
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE * 0.6)(x)  
    
    # Classification head
    outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Restore mixed precision policy if needed
    if Config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    return model
'''

import tensorflow as tf
from config import Config

def build_model() -> tf.keras.Model:
    """Build optimized CNN model using config parameters"""
    inputs = tf.keras.Input(shape=(*Config.Training.IMAGE_SIZE, 3))
    
    x = inputs
    for filters in Config.Model.CONV_FILTERS:
        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(Config.Model.L2_LAMBDA))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        Config.Model.DENSE_UNITS, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(Config.Model.L2_LAMBDA))(x)
    x = tf.keras.layers.Dropout(Config.Model.DROPOUT_RATE)(x)
     
    outputs = tf.keras.layers.Dense(
        Config.Training.NUM_CLASSES, 
        activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)