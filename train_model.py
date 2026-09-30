import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Global configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
CLASS_NAMES = ['benign', 'invalid', 'malignant']
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_model.keras")

def build_model(input_shape=(128, 128, 3), num_classes=3):
    """
    Builds a robust, lightweight 4-stage CNN with Batch Normalization
    and Global Average Pooling for histopathology image classification.
    
    Key Design Benefits:
      1. Batch Normalization: Stabilizes activations against H&E staining variations.
      2. Global Average Pooling: Eliminates parameter explosion (reduces params from 4.2M to 250k),
         drastically reducing overfitting while improving inference speed on CPU/device.
      3. Multi-tier Dropout & L2 Regularization: Enforces robust feature representations.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Block 1: 128x128 -> 64x64
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.10)(x)
    
    # Block 2: 64x64 -> 32x32
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)
    
    # Block 3: 32x32 -> 16x16
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)
    
    # Block 4: 16x16 -> 8x8
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Classifier Head with Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="BreastCancerHistopathologyCNN")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_data_augmentation():
    """
    Data augmentation for morphological invariance across orientations and minor zooms.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
    ])

def train_and_evaluate():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory {train_dir} not found. Run prepare_data.py first.")
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading datasets...")
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
    
    # Compute balanced class weights to eliminate majority-class bias
    labels_list = []
    for _, y in train_ds:
        labels_list.extend(y.numpy())
    labels_arr = np.array(labels_list)
    weights = compute_class_weight('balanced', classes=np.unique(labels_arr), y=labels_arr)
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f"Balanced Class Weights: {class_weight_dict}")
    
    # Preprocessing / Normalization pipeline
    normalization_layer = layers.Rescaling(1./255)
    augmentation = get_data_augmentation()
    
    # Apply augmentation + normalization to train
    train_ds_proc = train_ds.map(lambda x, y: (normalization_layer(augmentation(x, training=True)), y),
                                 num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    # Apply normalization only to validation & test
    val_ds_proc = val_ds.map(lambda x, y: (normalization_layer(x), y),
                             num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds_proc = test_ds.map(lambda x, y: (normalization_layer(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    print("\nInitializing CNN model...")
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=len(class_names))
    model.summary()
    
    training_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    ]
    
    print(f"\nStarting training for up to {EPOCHS} epochs with balanced class weighting...")
    history = model.fit(
        train_ds_proc,
        validation_data=val_ds_proc,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=training_callbacks
    )
    
    # Save the final best model
    model.save(MODEL_PATH)
    print(f"\nSaved best model to: {MODEL_PATH}")
    
    # Evaluate on unseen Test Set
    print("\n=============================================")
    print("           TEST SET EVALUATION               ")
    print("=============================================")
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds_proc:
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
        
    return model

if __name__ == "__main__":
    train_and_evaluate()

