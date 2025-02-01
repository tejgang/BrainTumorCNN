from model import build_model
from data_loading import load_data
from visual import plot_training_history
from config import Config
from dir import Dir
import tensorflow as tf

# Focal Loss implementation for handling class imbalance
# gamma: Focusing parameter that reduces the loss contribution from easy examples
# alpha: Weighting factor for rare classes
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        # Calculate pt_1 where predictions match ground truth
        # For positive samples (y_true=1), pt_1 = predicted prob
        # For negative samples (y_true=0), pt_1 = 1
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        
        # Focal loss formula: -α(1-pt)ᵧ log(pt)
        # This reduces the impact of well-classified examples and focuses on hard ones
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + tf.keras.backend.epsilon()))
    return focal_loss_fixed

def train_model():
    # Enable mixed precision training for better memory efficiency and speed
    # Uses float16 for certain operations while maintaining float32 for critical ones
    if Config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Initialize data generators and model architecture
    train_generator, validation_generator, _ = load_data()
    model = build_model()
    
    # Model compilation with optimized settings
    model.compile(
        # Adam optimizer with configurable learning rate
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        # Focal loss for handling class imbalance
        loss=focal_loss(gamma=2.0),
        # Multiple metrics for comprehensive model evaluation
        metrics=['accuracy',  # Standard accuracy
                tf.keras.metrics.AUC(name='auc'),  # Area Under ROC Curve
                tf.keras.metrics.Precision(name='precision'),  # True positives / (True + False positives)
                tf.keras.metrics.Recall(name='recall'),  # True positives / (True + False negatives)
                tf.keras.metrics.F1Score(average='macro', name='f1_macro')]  # Harmonic mean of precision and recall
    )
    
    # Training callbacks for optimization and monitoring
    callbacks = [
        # Early Stopping: Prevents overfitting by monitoring validation metrics
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_macro',  # Monitor macro-averaged F1 score
            patience=Config.EARLY_STOPPING_PATIENCE,  # Number of epochs to wait for improvement
            restore_best_weights=True,  # Restore model to best performance
            mode='max',  # Higher F1 is better
            verbose=1
        ),
        # Learning Rate Reduction: Adapts learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_macro',
            factor=Config.REDUCE_LR_FACTOR,  # Factor to reduce learning rate by
            patience=Config.REDUCE_LR_PATIENCE,  # Epochs to wait before reduction
            min_lr=1e-8,  # Minimum learning rate
            mode='max',
            verbose=1
        ),
        # Model Checkpointing: Saves best model during training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=Dir.MODEL_SAVE_PATH,
            monitor='val_f1_macro',
            save_best_only=True,  # Only save when model improves
            mode='max',
            verbose=1
        )
    ]
    
    # Train the model with optimized parameters
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=Config.CLASS_WEIGHTS  # Apply class weights to handle imbalance
    )

    # Visualize and save training progress
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


