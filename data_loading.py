from config import Config
from dir import Dir
import tensorflow as tf

def load_data():
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        Dir.TRAIN_DIR,
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        Dir.TRAIN_DIR,
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Test data (no augmentation)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        Dir.TEST_DIR,
        target_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator