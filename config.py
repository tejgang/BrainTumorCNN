class Config:
    # Model training parameters
    IMAGE_SIZE = (224, 224)  # Reduced for speed
    BATCH_SIZE = 64         # Increased for speed
    EPOCHS = 50
    LEARNING_RATE = 1e-5
    NUM_CLASSES = 4
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.2
    
    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5
    L2_LAMBDA = 0.01
    
    # Performance optimization
    MIXED_PRECISION = True
    NUM_WORKERS = 4
    USE_MULTIPROCESSING = True
    MAX_QUEUE_SIZE = 10
    
   
    