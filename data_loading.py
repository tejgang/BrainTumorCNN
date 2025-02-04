from config import Config
from dir import Dir
import tensorflow as tf

def load_data():

    def _load_dataset(directory, subset=None, augment=False):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=Config.IMAGE_SIZE,
            batch_size=Config.BATCH_SIZE,
            validation_split=Config.VALIDATION_SPLIT if subset else None,
            subset=subset,
            seed=42
        )
        
        # Convert labels to one-hot encoding
        def one_hot(image, label):
            return image, tf.one_hot(label, Config.NUM_CLASSES)
            
        # Preprocess images and convert labels to one-hot
        ds = ds.map(
            lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augment:
            # Data augmentation using available functions
            ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), 
                       num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (tf.image.random_flip_up_down(x), y), 
                       num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (tf.image.random_brightness(x, 0.1), y), 
                       num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (tf.image.random_contrast(x, 0.9, 1.1), y), 
                       num_parallel_calls=tf.data.AUTOTUNE)
        
        return ds.prefetch(tf.data.AUTOTUNE).cache()

    train_ds = _load_dataset(Dir.TRAIN_DIR, subset='training', augment=True)
    val_ds = _load_dataset(Dir.TRAIN_DIR, subset='validation')
    test_ds = _load_dataset(Dir.TEST_DIR)
    
    return train_ds, val_ds, test_ds
    