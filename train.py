from model import build_model
from data_loading import load_data
from visual import plot_training_history
from config import Config
from dir import Dir
import tensorflow as tf
import numpy as np

# Custom loss combining focal loss with class weighting for tumor classification.
def weighted_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, class_weights=None):
    
    # Convert to float32
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate cross entropy
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Calculate focal term
    pt = tf.exp(-ce)
    focal_term = tf.pow(1 - pt, gamma)
    
    # Apply class weights
    if class_weights is not None:
        weighted_ce = y_true * tf.cast(class_weights, tf.float32) * ce
    else:
        weighted_ce = y_true * ce
    
    # Combine focal loss with class weights
    loss = focal_term * weighted_ce
    return tf.reduce_mean(loss)

# Get class distribution from data generator
def get_class_distribution(data_generator):
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    num_samples = data_generator.n
    num_batches = num_samples // data_generator.batch_size + (0 if num_samples % data_generator.batch_size == 0 else 1)
    
    for i in range(num_batches):
        _, y = data_generator.__getitem__(i)
        for label in y:
            class_idx = np.argmax(label)
            class_counts[class_idx] += 1
    
    counts_list = [class_counts[i] for i in range(4)]
    return class_counts, counts_list

def train_model():

    # Load data and model
    train_generator, validation_generator, _ = load_data()
    
    # Get class distribution
    _, train_counts = get_class_distribution(train_generator)
    
    # Calculate class weights based on training distribution
    total = sum(train_counts)
    n_classes = len(train_counts)
    class_weights = [total/(n_classes*count) if count > 0 else 0 for count in train_counts]
    
    # Build and compile model with calculated weights
    model = build_model()
    model.compile(
        loss=lambda y_true, y_pred: weighted_focal_loss(y_true, y_pred, class_weights=class_weights),
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(Config.MODEL_SAVE_PATH, save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Save training plots
    plot_training_history(history, Config.PLOT_SAVE_PATH)

if __name__ == "__main__":
    train_model()


