from config import Config
from dir import Dir
import tensorflow as tf

def load_data():
    # Enhanced data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,        
        width_shift_range=0.3,    
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,          
        horizontal_flip=True,
        vertical_flip=True,      
        brightness_range=[0.7,1.3],  
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
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