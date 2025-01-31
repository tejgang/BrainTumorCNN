import numpy as np
import tensorflow as tf
from data_loading import load_data
from dir import Dir
from visual import generate_confusion_matrix

def evaluate_model():
    # Load data and model (using .keras format)
    _, _, test_generator = load_data()
    try:
        model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH)
    except:
        # Fallback to old path if needed
        old_path = Dir.MODEL_SAVE_PATH.replace('.keras', '.h5')
        model = tf.keras.models.load_model(old_path)
        print(f"Loaded model from old format: {old_path}")

    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = []
    for _, labels in test_generator:
        y_true.extend(np.argmax(labels, axis=1))
    y_true = np.array(y_true[:len(y_pred)])  # Match lengths
    
    # Evaluate model
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate and save confusion matrix
    generate_confusion_matrix(y_true, y_pred, Dir.CONFUSION_MATRIX_SAVE_PATH)

if __name__ == "__main__":
    evaluate_model()    
