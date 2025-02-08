import numpy as np
import tensorflow as tf
from data_loading import load_data
from dir import Dir
from visual import generate_confusion_matrix
from config import Config

def evaluate_model():
    """Evaluate the trained model on the test dataset"""
    # Load test data
    _, _, test_ds = load_data()
    
    # Load trained model
    model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH)
    
    print("Starting evaluation...")
    
    # Collect predictions and true labels
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        # Predict in batches for efficiency
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))
    
    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Evaluate model metrics
    print("\nCalculating metrics...")
    metrics = model.evaluate(test_ds, verbose=1)
    
    # Print evaluation metrics
    metric_names = ['loss', 'accuracy', 'auc']
    print("\nEvaluation Metrics:")
    for name, value in zip(metric_names, metrics):
        print(f"{name.capitalize()}: {value:.4f}")
    
    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, Dir.CONFUSION_MATRIX_SAVE_PATH)

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(Config.Optimization.SEED)
    
    # Set mixed precision if enabled
    if Config.Optimization.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    evaluate_model()