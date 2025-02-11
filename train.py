from model import build_model
from data_loading import load_data
from visual import plot_training_history
from config import Config
from dir import Dir
import tensorflow as tf


def train_model():
    # Load data
    train_ds, val_ds, _ = load_data()
    
    # Build model
    model = build_model()
    
    # Model compilation with optimized settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.F1Score(average='macro', name='f1_macro')]
    )
    
    # Callbacks
    callbacks = [
        # Early Stopping: Prevents overfitting by monitoring validation metrics
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True
        ),

        # Learning Rate Reduction: Adapts learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.2,
            patience=3
        )
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks,

    )

    # Visualize and save training progress
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


