class Config:

    # Dataset paths
    TRAIN_DIR = "C:/Users/khila/Downloads/archive/Training"
    TEST_DIR = "C:/Users/khila/Downloads/archive/Testing"
        
    # Model hyperparameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    NUM_CLASSES = 4  # 3 tumor types + no tumor
        
    # Save paths
    MODEL_SAVE_PATH = "saved_model/brain_tumor_model.h5"
    PLOT_SAVE_PATH = "results/training_plot.png"