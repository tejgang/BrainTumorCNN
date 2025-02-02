class Config:
    # Model training parameters
    IMAGE_SIZE = (224, 224)  
    BATCH_SIZE = 64         
    EPOCHS = 50           
    LEARNING_RATE = 2e-5    
    NUM_CLASSES = 4
    
    # Class weights (adjusted based on confusion matrix)
    CLASS_WEIGHTS = {
        0: 2.0,    # No Tumor (increased due to Glioma confusion)
        1: 3.5,    # Glioma (increased due to most misclassifications)
        2: 1.0,    # Meningioma (performing well, baseline weight)
        3: 2.5     # Pituitary (moderate weight due to Glioma confusion)
    }
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 5    
    REDUCE_LR_PATIENCE = 3         
    REDUCE_LR_FACTOR = 0.15        
    

    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5           
    L2_LAMBDA = 0.001           
    
    # Performance optimization
    MIXED_PRECISION = True
    USE_MULTIPROCESSING = True
    MAX_QUEUE_SIZE = 20
    NUM_WORKERS = 8
    
   
    