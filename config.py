class Config:
    # Model training parameters
    BATCH_SIZE = 64  
    IMAGE_SIZE = (256, 256)  
    EPOCHS = 50  

    # Paths for saving outputs
    MODEL_SAVE_PATH = './saved_models/model.h5'
    PLOT_SAVE_PATH = './plots/training_history.png'