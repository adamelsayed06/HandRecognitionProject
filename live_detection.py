import tensorflow as tf
import numpy as np
import cv2
import pyautogui

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Define image size (must match the size used for training)
IMG_SIZE = 224

# Function to preprocess each frame from webcam
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(frame_resized, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to [0,1]
    return img_array

# Function to simulate key press based on prediction
def simulate_key_press(prediction):
    if prediction >= 0.5:
        print("Detected Palm, pressing UP arrow")
        pyautogui.press('up')
    else:
        print("Detected Fist, pressing DOWN arrow")
        pyautogui.press('down')

# OpenCV to capture live video feed

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    img_array = preprocess_frame(frame)

    # Get the prediction from the model
    prediction = model.predict(img_array)[0][0]

    # Simulate key press based on prediction
    simulate_key_press(prediction)

    # Display prediction on the video feed
    label = "Palm" if prediction >= 0.5 else "Fist"
    color = (0, 255, 0) if prediction >= 0.5 else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Live Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
