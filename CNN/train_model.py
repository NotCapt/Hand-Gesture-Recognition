import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
import os
import numpy as np

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 4  # down, left, right, up
SEED = 42

# Paths
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/valid'



# Count images in each directory
for split in ['train', 'valid']:
    split_dir = f'dataset/{split}'
    total = 0
    for class_dir in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
            print(f"{split}/{class_dir}: {count} images")
            total += count
    print(f"Total {split}: {total} images")
print("=" * 60)

# Load datasets 
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',  # One-hot encoding
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    seed=SEED
)

# Print class names
class_names = train_ds.class_names
print("\nClass names:", class_names)

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)

# Apply normalization
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))


# Data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    # Input layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # Data augmentation 
    data_augmentation,
    
    # First Conv Block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Conv Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fourth Conv Block
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    
    # Flatten
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout for regularization
    # layers.Dense(256, activation='relu'),
    # layers.Dropout(0.5),
    
    # Output Layer
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("\nModel Architecture:")
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'gesture_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('gesture_model_final.h5')
print("\n Model saved as 'gesture_model_final.h5'")
print(" Best model saved as 'gesture_model_best.h5'")

# Final evaluation
print("\n" + "=" * 60)
print("Final Evaluation")
print("=" * 60)
val_loss, val_accuracy = model.evaluate(valid_ds)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Training summary
print("Training summary")
best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
print(f"Best epoch: {best_epoch + 1}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
