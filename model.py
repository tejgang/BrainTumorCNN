import tensorflow as tf
from config import Config


def build_model():
    model = tf.keras.Sequential([

        # Input layer
        tf.keras.layers.Input(shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)),
    
        # Convolutional layer 1
        tf.keras.layers.Conv2D(32, (4, 4), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Dropout(0.25),

        # Convolutional layer 2
        tf.keras.layers.Conv2D(64, (4, 4), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Dropout(0.25),

        # Convolutional layer 3
        tf.keras.layers.Conv2D(128, (4, 4), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Dropout(0.25),

        # Convolutional layer 4
        tf.keras.layers.Conv2D(128, (4, 4), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),

        # Full connect layers
        tf.keras.layers.Dense(512, activation="relu", 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation="softmax")
    ])

    
    return model
