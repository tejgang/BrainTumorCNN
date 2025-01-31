import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from data_loading import load_data
from dir import Dir
#from visual import plot_confusion_matrix

def evaluate_model():
    # Load data and model
    _, _, test_generator = load_data()
    model = tf.keras.models.load_model(Dir.MODEL_SAVE_PATH)

    # Get class names
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    class_indices = {name: i for i, name in enumerate(class_names)}

    # Reset generator before predictions
    test_generator.reset()
    
    # Generate predictions for all test data
    num_samples = test_generator.samples
    steps = int(np.ceil(num_samples / test_generator.batch_size))  # Convert to int
    
    predictions = model.predict(test_generator, steps=steps)
    y_pred_classes = np.argmax(predictions, axis=1)
    
    # Reset generator to get true labels
    test_generator.reset()
    
    # Get true labels
    y_true = []
    for i in range(steps):  # steps is now an int
        _, labels = test_generator[i]
        y_true.extend(np.argmax(labels, axis=1))
    y_true = np.array(y_true[:num_samples])  # Trim to actual number of samples
    
    # Evaluate on test data
    test_generator.reset()
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Create figure and plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    plt.savefig(Dir.CONFUSION_MATRIX_SAVE_PATH)
    print(f"Confusion matrix saved to: {Dir.CONFUSION_MATRIX_SAVE_PATH}")
    plt.close()

if __name__ == "__main__":
    evaluate_model()    
