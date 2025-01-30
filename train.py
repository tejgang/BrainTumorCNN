from model import build_model
from data_loading import load_data
from visual import plot_training_history
from config import Config
from dir import Dir
import tensorflow as tf


def train_model():
    # Load data and model
    train_generator, validation_generator, _ = load_data()
    model = build_model()

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(Dir.MODEL_SAVE_PATH, save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Save training plots
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


