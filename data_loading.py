from config import Config
from dir import Dir
import tensorflow as tf


def load_data():
   
   # Data Augmentation
   train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   brightness_range=(0.85, 1.15),
                                   width_shift_range=0.002,
                                   height_shift_range=0.002,
                                   shear_range=12.5,
                                   zoom_range=0,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   fill_mode="nearest")


# applying the generator to training data with constant seed
   train_generator = train_datagen.flow_from_directory(Dir.TRAIN_DIR,
                                                    target_size=Config.IMAGE_SIZE,
                                                    batch_size=Config.BATCH_SIZE,
                                                    class_mode="categorical",
                                                    seed=Config.SEED)

# No augmentation of the test data, just rescaling
   test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# applying the generator to testing data with constant seed
   test_generator = test_datagen.flow_from_directory(Dir.TEST_DIR,
                                                  target_size=Config.IMAGE_SIZE,
                                                  batch_size=Config.BATCH_SIZE,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed=Config.SEED)
   
   return train_generator, test_generator
    