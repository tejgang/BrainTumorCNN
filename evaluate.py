import numpy as np
import tensorflow as tf
from data_loading import load_data
from dir import Dir
from visual import generate_confusion_matrix
from train import focal_loss  

def evaluate_model():
    # Load data and model
    _, _, test_generator = load_data()
    
    # Load model with custom loss
    custom_objects = {
        'focal_loss_fixed': focal_loss()
    }
    model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH, custom_objects=custom_objects)

    # Calculate steps as integer
    steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    
    print("Starting evaluation...")
    
    # Get predictions with steps
    predictions = model.predict(
        test_generator,
        steps=steps,
        verbose=1
    )
    y_pred = np.argmax(predictions, axis=1)
    
    # Reset generator
    test_generator.reset()
    
    # Get true labels
    y_true = test_generator.classes[:len(y_pred)]
    
    # Evaluate model with steps
    print("\nCalculating metrics...")
    metrics = model.evaluate(
        test_generator,
        steps=steps,
        verbose=1
    )
    
    # Print all metrics
    metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    for name, value in zip(metric_names, metrics):
        print(f"\nTest {name.capitalize()}: {value:.4f}")

    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, Dir.CONFUSION_MATRIX_SAVE_PATH)

if __name__ == "__main__":
    evaluate_model()    
