class Dir:
    # Default relative paths
    TRAIN_DIR = "./data/Training"
    TEST_DIR = "./data/Testing"

    try:
        from .local_paths import TRAIN_DIR, TEST_DIR  # Import specific paths
    except ImportError:
        pass