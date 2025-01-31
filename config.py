class Config:
    # Model training parameters
    IMAGE_SIZE = (224, 224)  
    BATCH_SIZE = 32        
    EPOCHS = 100           
    LEARNING_RATE = 1e-5   
    NUM_CLASSES = 4
    
    # Class weights (based on confusion matrix analysis)
    CLASS_WEIGHTS = {
        0: 1.0,    # No Tumor
        1: 3.0,    # Glioma (increased weight)
        2: 1.0,    # Neurinoma
        3: 4.0     # Pituitary (highest weight)
    }
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    REDUCE_LR_FACTOR = 0.1
    
    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5
    L2_LAMBDA = 0.001
    
    # Performance optimization
    MIXED_PRECISION = True
    NUM_WORKERS = 4
    USE_MULTIPROCESSING = True
    MAX_QUEUE_SIZE = 10
    
   
    