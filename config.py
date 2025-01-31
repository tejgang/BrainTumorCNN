class Config:
    # Model training parameters
    IMAGE_SIZE = (224, 224) 
    BATCH_SIZE = 32         
    EPOCHS = 30            
    LEARNING_RATE = 1e-4    
    NUM_CLASSES = 4         # 3 tumor types + no tumor

    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 5    # Reduced from 10 to 5
    REDUCE_LR_PATIENCE = 3         # Kept at 3 for quick LR adjustments
    REDUCE_LR_FACTOR = 0.2
    
    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5
    L2_LAMBDA = 0.01

   
    