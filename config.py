class Config:
    # Improved training parameters
    IMAGE_SIZE = (224, 224)  # Increased size for better detail
    BATCH_SIZE = 32         # Smaller batch size for better generalization
    EPOCHS = 100           # More epochs with early stopping
    LEARNING_RATE = 1e-4   # Lower learning rate for stability
    
    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5
    L2_LAMBDA = 0.01

    # Model training parameters
    NUM_CLASSES = 4  # 3 tumor types + no tumor