import tensorflow as tf
from config import Config

def build_model():
    # Use a pre-trained model as base
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)
    )
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
        layer.trainable = True
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA)),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_LAMBDA)),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    return model
