from local_paths import train_dir, test_dir
from local_paths import training_plots_dir, confusion_matrix_dir

class Dir:
    # Data paths
    TRAIN_DIR = train_dir
    TEST_DIR = test_dir

    # Saved models and plots
    # Paths for saving outputs
    MODEL_SAVE_PATH = './saved_models/model.h5'
    PLOT_SAVE_PATH = training_plots_dir
    CONFUSION_MATRIX_SAVE_PATH = confusion_matrix_dir