import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define image size and batch size
IMG_SIZE = 224  # VGG16 expects images of size 224x224
BATCH_SIZE = 32

# 1. Load the VGG16 model, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# 2. Freeze the base model (we don’t want to modify the pretrained weights in early stages)
base_model.trainable = False

# 3. Add custom layers for binary classification (palm vs. fist)
model = models.Sequential([
    base_model,
    layers.Flatten(),  # Flatten the output of the convolutional layers
    layers.Dense(128, activation='relu'),  # Add a fully connected layer
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# 4. Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  # Binary crossentropy for binary classification
              metrics=['accuracy'])

# 5. Set up ImageDataGenerator for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip images horizontally
    validation_split=0.2  # Reserve 20% of data for validation
)

# 6. Load training and validation data
train_generator = train_datagen.flow_from_directory(
    'dataset/',  # Path to dataset
    target_size=(IMG_SIZE, IMG_SIZE),  # Resize images to match VGG16 input size
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification (palm or fist)
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 7. Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Set the number of epochs
    validation_data=validation_generator
)

# 8. Fine-tune the model (optional)
# Unfreeze the base_model and fine-tune it
base_model.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training with a low learning rate to fine-tune the VGG16 layers
history_fine = model.fit(
    train_generator,
    epochs=10,  # Additional epochs for fine-tuning
    validation_data=validation_generator
)