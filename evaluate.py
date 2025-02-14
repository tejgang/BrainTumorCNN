import numpy as np
import tensorflow as tf
from data_loading import load_data
from dir import Dir
from visual import generate_confusion_matrix

def evaluate_model():
    # Load data and model
    _, _, test_ds = load_data()
    model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH)
    
    print("Starting evaluation...")
    
    # Get predictions and true labels
    y_pred = []
    y_true = []
    
    # Iterate through the dataset
    for images, labels in test_ds:
        # Get predictions for this batch
        predictions = model.predict(images, verbose=0)
        # Convert predictions and labels to class indices
        pred_indices = np.argmax(predictions, axis=1)
        true_indices = np.argmax(labels, axis=1)
        
        y_pred.extend(pred_indices)
        y_true.extend(true_indices)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Evaluate model
    print("\nCalculating metrics...")
    metrics = model.evaluate(test_ds, verbose=1)
    
    # Print all metrics
    metric_names = ['loss', 'accuracy', 'auc']
    for name, value in zip(metric_names, metrics):
        print(f"\nTest {name.capitalize()}: {value:.4f}")

    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, Dir.CONFUSION_MATRIX_SAVE_PATH)

if __name__ == "__main__":
    evaluate_model()    
