class Config:
    # Model training parameters
    IMAGE_SIZE = (224, 224)  # Keep full resolution for better feature detection
    BATCH_SIZE = 32         # Smaller batch size for better generalization
    EPOCHS = 100           
    LEARNING_RATE = 2e-5    # Slightly increased for better exploration
    NUM_CLASSES = 4
    
    # Class weights (adjusted based on confusion matrix)
    CLASS_WEIGHTS = {
        0: 2.0,    # No Tumor (increased due to Glioma confusion)
        1: 3.5,    # Glioma (increased due to most misclassifications)
        2: 1.0,    # Meningioma (performing well, baseline weight)
        3: 2.5     # Pituitary (moderate weight due to Glioma confusion)
    }
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 12    # Increased to allow more training
    REDUCE_LR_PATIENCE = 6         # Adjusted accordingly
    REDUCE_LR_FACTOR = 0.15        # More gradual learning rate reduction
    
    # Data augmentation parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.65           # Increased to combat overfitting
    L2_LAMBDA = 0.002            # Increased regularization
    
    # Performance optimization
    MIXED_PRECISION = True
    USE_MULTIPROCESSING = True
    MAX_QUEUE_SIZE = 10
    NUM_WORKERS = 4
    
   
    