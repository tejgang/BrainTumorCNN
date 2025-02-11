from config import Config
from dir import Dir
import tensorflow as tf


def load_data():
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])
    
    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Dir.TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=Config.SEED,
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    
    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        Dir.TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=Config.SEED,
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    
    # Load test data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        Dir.TEST_DIR,
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    
    val_ds = val_ds.prefetch(AUTOTUNE).cache()
    test_ds = test_ds.prefetch(AUTOTUNE).cache()
    
    return train_ds, val_ds, test_ds
    