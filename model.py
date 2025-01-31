import tensorflow as tf
from config import Config

def build_model():
    # Use a pre-trained model as base
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    return model
