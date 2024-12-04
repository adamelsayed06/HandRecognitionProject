from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models
from pynput.keyboard import Key, Controller
import tensorflow as tf
import os
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('hand_model.h5')

# Initialize the keyboard
keyboard = Controller()

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use second camera, not sure why 0 doesn't work

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=(0, -1))

    # Make prediction
    prediction = model.predict(input_frame)
    class_id = np.argmax(prediction)

    if class_id == 0: #Palm
        keyboard.press(Key.up)
    else: #Fist
        keyboard.release(Key.down)

    cv2.imshow("Hand Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
