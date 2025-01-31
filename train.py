from model import build_model
from data_loading import load_data
from visual import plot_training_history
from config import Config
from dir import Dir
import tensorflow as tf


def train_model():
    # Enable mixed precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Load data and model
    train_generator, validation_generator, _ = load_data()
    model = build_model()
    
    # Learning rate schedule with proper initial learning rate
    initial_learning_rate = Config.LEARNING_RATE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    # Optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)  # Use fixed learning rate
    
    # Compile with weighted loss
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            Dir.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train with class weights
    class_weights = {
        0: 1.0,  # No Tumor
        1: 2.0,  # Glioma
        2: 2.0,  # Meningioma
        3: 2.0   # Pituitary
    }
    
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Save training plots
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


