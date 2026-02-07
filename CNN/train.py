import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from utils import IMG_SIZE

# Configuration
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = 'dataset' 
NUM_CLASSES = 4

def create_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Load Data
print("Loading datasets...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/train',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/valid',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

# Normalize [0,1]
norm_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (norm_layer(x), y))
valid_ds = valid_ds.map(lambda x, y: (norm_layer(x), y))

# Train
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

cbs = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('gesture_model_best.h5', save_best_only=True)
]

history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=cbs)
model.save('gesture_model_final.h5')
print("Training complete.")