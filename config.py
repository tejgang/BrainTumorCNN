class Config:
    class Training:
        IMAGE_SIZE = (150, 150)
        BATCH_SIZE = 32
        EPOCHS = 30
        LEARNING_RATE = 2e-5
        NUM_CLASSES = 4
        VALIDATION_SPLIT = 0.2
        TEST_SPLIT = 0.1

    class EarlyStopping:
        PATIENCE = 5
        REDUCE_LR_PATIENCE = 3
        REDUCE_LR_FACTOR = 0.15

    class Model:
        DROPOUT_RATE = 0.5
        L2_LAMBDA = 0.001
        CONV_FILTERS = [32, 64, 128]
        DENSE_UNITS = 128

    class Optimization:
        MIXED_PRECISION = True
        SEED = 42
    
   
    