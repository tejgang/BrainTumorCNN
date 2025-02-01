import tensorflow as tf
from config import Config

def attention_block(x):
    channels = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(channels // 8, activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    return x * tf.keras.layers.Reshape((1, 1, channels))(se)

def build_model():
    # Use a pre-trained model as base
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)
    )
    
    # Unfreeze more layers for better feature extraction
    for layer in base_model.layers[-60:]:
        layer.trainable = True
    
    inputs = tf.keras.Input(shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3))
    x = base_model(inputs)
    
    # Add attention mechanism
    x = attention_block(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(1024, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA))(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA))(x)
    x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
    
    outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
