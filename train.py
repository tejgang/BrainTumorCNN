
'''
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
'''

from model import build_model
from data_loading import load_data
from config import Config
from dir import Dir
from visual import plot_training_history
import tensorflow as tf

def get_class_weights(dataset: tf.data.Dataset) -> dict:
    """Calculate class weights using TF operations"""
    class_counts = tf.zeros((Config.Training.NUM_CLASSES,), dtype=tf.int32)
    for _, labels in dataset:
        class_counts += tf.math.bincount(tf.argmax(labels, axis=1), 
                                       minlength=Config.Training.NUM_CLASSES)
    
    total = tf.reduce_sum(class_counts)
    return {i: float(total / (Config.Training.NUM_CLASSES * count)) 
            for i, count in enumerate(class_counts.numpy())}

def train_model():
    train_ds, val_ds, _ = load_data()
    
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.Training.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.Training.EPOCHS,
        class_weight=get_class_weights(train_ds),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=Config.EarlyStopping.PATIENCE,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=Config.EarlyStopping.REDUCE_LR_FACTOR,
                patience=Config.EarlyStopping.REDUCE_LR_PATIENCE,
                mode='max'
            )
        ]
    )

    # Save the model
    model.save(Dir.MODEL_SAVE_PATH)
    
    # Visualize and save training progress
    plot_training_history(history, Dir.PLOT_SAVE_PATH)

    return history

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(Config.Optimization.SEED)
    if Config.Optimization.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    train_model()

