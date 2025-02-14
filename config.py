import tensorflow as tf
import numpy as np
import random

class Config:
    # Model training parameters
    IMAGE_SIZE = (150, 150)  
    BATCH_SIZE = 32         
    EPOCHS = 40
    NUM_CLASSES = 4
    SEED = 42  
    LEARNING_RATE = 0.0005  

    # Set all random seeds for reproducibility
    @staticmethod
    def set_seeds():
        tf.random.set_seed(Config.SEED)
        np.random.seed(Config.SEED)
        random.seed(Config.SEED)
        
        # Set operation-level seed for even more reproducibility
        tf.keras.utils.set_random_seed(Config.SEED)
        
        # Configure operation determinism
        tf.config.experimental.enable_op_determinism()

# Set seeds when module is imported
Config.set_seeds()

    
    
    