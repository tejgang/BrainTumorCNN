import tensorflow as tf


def build_model():
    model = tf.keras.models.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(160, 160, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')  
    ])
    
    return model
