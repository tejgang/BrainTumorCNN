import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from data_loading import load_data
from dir import Dir
from visual import plot_confusion_matrix

def evaluate_model():

    # Load data and model
    _, _, test_generator = load_data()
    model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH)

    # Evaluate on test data
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, test_generator.class_indices)

    # Save plots
    plt.savefig(Dir.CONFUSION_MATRIX_SAVE_PATH)
    plt.close()

if __name__ == "__main__":
    evaluate_model()    
