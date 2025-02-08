import tensorflow as tf
from typing import Tuple
from config import Config
from dir import Dir

def apply_augmentation(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies random augmentations to image"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return tf.clip_by_value(image, 0.0, 1.0), label

def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Standard preprocessing pipeline"""
    return (tf.cast(image, tf.float32) / 255.0, 
            tf.one_hot(label, Config.Training.NUM_CLASSES))

def create_dataset(data_dir: str, subset: str = None, do_augment: bool = False) -> tf.data.Dataset:
    """Create optimized dataset pipeline"""
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        subset=subset,
        validation_split=Config.Training.VALIDATION_SPLIT if subset else None,
        seed=Config.Optimization.SEED,
        image_size=Config.Training.IMAGE_SIZE,
        batch_size=Config.Training.BATCH_SIZE
    )
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if do_augment:
        ds = ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        
    return ds.prefetch(tf.data.AUTOTUNE)

def load_data():
    """Load all datasets with proper configurations"""
    return (
        create_dataset(Dir.TRAIN_DIR, 'training', do_augment=True),
        create_dataset(Dir.TRAIN_DIR, 'validation'),
        create_dataset(Dir.TEST_DIR)
    )