import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

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
    
    # Save with full path
    try:
        plt.savefig(save_path)
        print(f"Training plots saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

def generate_confusion_matrix(y_true, y_pred, save_path):
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Define class names
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='YlOrRd',  # Softer color palette
                cbar_kws={'label': 'Number of Predictions'},
                annot_kws={'size': 12},
                square=True)
    
    plt.title('Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
    finally:
        plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))