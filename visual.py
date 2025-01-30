import matplotlib.pyplot as plt
import seaborn as sns
import os
from dir import Dir

def plot_training_history(history, save_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    # Save with full path and ensure directory exists
    try:
        plt.savefig(save_path)
        print(f"Training plots saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

def plot_confusion_matrix(cm, class_indices):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(Dir.CONFUSION_MATRIX_SAVE_PATH), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_indices.keys(), 
                yticklabels=class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    try:
        plt.savefig(Dir.CONFUSION_MATRIX_SAVE_PATH)
        print(f"Confusion matrix saved to: {Dir.CONFUSION_MATRIX_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
    finally:
        plt.close()