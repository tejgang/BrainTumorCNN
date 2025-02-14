import numpy as np
import tensorflow as tf
from data_loading import load_data
from dir import Dir
from visual import generate_confusion_matrix

def evaluate_model():
    # Load data and model
    _, test_generator = load_data()
    model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH)
    
    print("Starting evaluation...")
    
    # Get predictions
    y_pred = []
    y_true = []
    
    # Get the number of steps
    steps = test_generator.samples // test_generator.batch_size
    if test_generator.samples % test_generator.batch_size != 0:
        steps += 1
    
    print(f"Total evaluation steps: {steps}")
    
    # Use model.predict on the generator
    predictions = model.predict(test_generator, steps=steps, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes[:len(y_pred)]
    
    # Evaluate model
    print("\nCalculating metrics...")
    test_generator.reset()
    metrics = model.evaluate(test_generator, steps=steps, verbose=1)
    
    # Print all metrics
    metric_names = ['loss', 'accuracy', 'auc']
    for name, value in zip(metric_names, metrics):
        print(f"\nTest {name.capitalize()}: {value:.4f}")

    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, Dir.CONFUSION_MATRIX_SAVE_PATH)

if __name__ == "__main__":
    evaluate_model()    
