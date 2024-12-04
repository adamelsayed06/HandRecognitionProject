from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models
import os
import cv2
import numpy as np

fist_folder = "/Users/adamelsayed/Downloads/archive/train/train/11"
palm_folder = "/Users/adamelsayed/Downloads/archive/train/train/5"

def load_preprocessed_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.endswith(('.jpg')):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)
    return images, labels

fist_images, fist_labels = load_preprocessed_images(fist_folder, 0)  # Label fists as 0
palm_images, palm_labels = load_preprocessed_images(palm_folder, 1)  # Label palms as 1

images = np.array(fist_images + palm_images)
images = np.expand_dims(images, axis=-1)  # Add channel dimension
labels = np.array(fist_labels + palm_labels)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two classes: fist, palm
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('hand_model.h5')
print("Model saved as 'hand_model.h5'")
