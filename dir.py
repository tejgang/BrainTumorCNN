from local_paths import train_dir, test_dir
from local_paths import training_plots_dir, confusion_matrix_dir

class Dir:
    # Data paths
    TRAIN_DIR = train_dir
    TEST_DIR = test_dir

    # Saved models and plots
    MODEL_SAVE_PATH = './saved_models/model.keras'
    WEIGHTS_SAVE_PATH = './saved_models/model.weights.h5'
    PLOT_SAVE_PATH = training_plots_dir
    CONFUSION_MATRIX_SAVE_PATH = confusion_matrix_dir