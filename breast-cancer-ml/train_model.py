import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# Global configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005
CLASS_NAMES = ['benign', 'invalid', 'malignant']
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_model.keras")

def build_model(input_shape=(128, 128, 3), num_classes=3):
    """
    Builds a simple, explainable 3-stage CNN for exhibition and viva presentation.
    Architecture:
      Input (128x128x3) -> Conv2D(32) -> ReLU -> MaxPool
      -> Conv2D(64) -> ReLU -> MaxPool
      -> Conv2D(128) -> ReLU -> MaxPool
      -> Flatten -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(3, Softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Stage 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Stage 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Stage 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_data_augmentation():
    """
    Training-only data augmentation pipeline.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
    ])

def train_and_evaluate():
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory {train_dir} not found. Run prepare_data.py first.")
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading datasets...")
    # Load raw datasets (without normalization yet, normalize via Rescaling layer or map)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Detected Classes: {class_names}")
    
    # Preprocessing / Normalization pipeline
    normalization_layer = layers.Rescaling(1./255)
    augmentation = get_data_augmentation()
    
    # Apply augmentation + normalization to train
    train_ds = train_ds.map(lambda x, y: (normalization_layer(augmentation(x, training=True)), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    # Apply normalization only to validation & test
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("\nInitializing CNN model...")
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=len(class_names))
    model.summary()
    
    training_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print(f"\nStarting training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=training_callbacks
    )
    
    # Save the final best model
    model.save(MODEL_PATH)
    print(f"\nSaved final model to: {MODEL_PATH}")
    
    # Evaluate on unseen Test Set
    print("\n=============================================\n           TEST SET EVALUATION\n=============================================")
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    test_acc = np.mean(y_true == y_pred) * 100
    print(f"Overall Test Accuracy: {test_acc:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"Rows: True {class_names}, Columns: Predicted {class_names}")
    print("=============================================")
    
    if test_acc >= 60.0:
        print(f"SUCCESS: Project test accuracy target achieved ({test_acc:.2f}% >= 60.0%)!")
    else:
        print(f"NOTE: Test accuracy {test_acc:.2f}% below 60% target.")

if __name__ == "__main__":
    train_and_evaluate()
