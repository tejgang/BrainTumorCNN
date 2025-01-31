class Config:
    # Model training parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50               # Increased epochs with early stopping
    LEARNING_RATE = 1e-5     # Lower learning rate for fine-tuning
    NUM_CLASSES = 4
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 10  # Increased patience
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.2
    
    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5
    L2_LAMBDA = 0.01        # L2 regularization factor
    
   
    