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
    
    # Model compilation 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,
            beta_1=0.9,  
            beta_2=0.999  
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    callbacks = [
        # Early Stopping: Prevents overfitting by monitoring validation metrics
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=5,     
            verbose=1,
            restore_best_weights=True 
        ),

        # Learning Rate Reduction: Adapts learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,     
            patience=5,
            min_lr=1e-6,    
            
        ),

        # Add ModelCheckpoint callback
        tf.keras.callbacks.ModelCheckpoint(
            filepath=Dir.MODEL_SAVE_PATH,
            monitor='val_loss',  
            save_best_only=True,
            mode='min',     
        )
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,  
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )

    # Visualize and save training progress
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


