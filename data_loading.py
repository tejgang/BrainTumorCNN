from config import Config
from dir import Dir
import tensorflow as tf


def load_data():
    def preprocess_image(image, label):
        # Convert to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        # Resize if needed
        image = tf.image.resize(image, Config.IMAGE_SIZE)
        
        # Data augmentation (only for training)
        return image, label

    def configure_for_performance(dataset, is_training=False):
        # Cache the dataset in memory
        dataset = dataset.cache()
        
        if is_training:
            # Shuffle and augment training data
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.map(
                lambda x, y: (tf.image.random_flip_left_right(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(
                lambda x, y: (tf.image.random_brightness(x, 0.2), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(
                lambda x, y: (tf.image.random_contrast(x, 0.8, 1.2), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch for performance
        dataset = dataset.batch(Config.BATCH_SIZE)
        return dataset.prefetch(tf.data.AUTOTUNE)

   

    # Load datasets using tf.data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Dir.TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=Config.SEED,
        image_size=Config.IMAGE_SIZE,
        batch_size=None,  # Load unbatched
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        Dir.TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=Config.SEED,
        image_size=Config.IMAGE_SIZE,
        batch_size=None,  # Load unbatched
        label_mode='categorical'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        Dir.TEST_DIR,
        seed=Config.SEED,
        image_size=Config.IMAGE_SIZE,
        batch_size=None,  # Load unbatched
        label_mode='categorical'
    )

    # Apply preprocessing and performance optimizations
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Configure datasets for performance
    train_ds = configure_for_performance(train_ds, is_training=True)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    return train_ds, val_ds, test_ds
    