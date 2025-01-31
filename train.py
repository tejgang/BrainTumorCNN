from model import build_model
from data_loading import load_data
from visual import plot_training_history
from config import Config
from dir import Dir
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + tf.keras.backend.epsilon()))
    return focal_loss_fixed

def train_model():
    # Enable mixed precision for faster training
    if Config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Load data and model
    train_generator, validation_generator, _ = load_data()
    model = build_model()
    
    # Compile with focal loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1')]
    )
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1',
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-8,
            mode='max',
            verbose=1
        ),
        # Save the entire model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=Dir.MODEL_SAVE_PATH,
            monitor='val_f1',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train with optimized parameters (removed workers and multiprocessing)
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=Config.CLASS_WEIGHTS,
        workers=Config.NUM_WORKERS,
        use_multiprocessing=Config.USE_MULTIPROCESSING
    )

    # Save training plots
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


