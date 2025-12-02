# test.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- 1. Configuration ---
MODEL_PATH = 'facialemotionmodel.h5'
IMAGE_SIZE = (48, 48)

# 🔴 🔴 🔴 UPDATE THIS BASED ON OUTPUT FROM bla.py! 🔴 🔴 🔴
# Example: if bla.py printed: {'happy': 0, 'neutral': 1}
# Then use: {0: 'happy', 1: 'neutral'}
#
# Common mappings:
# Option A:
# CLASS_INDICES = {0: 'happy', 1: 'neutral'}
#
# Option B (if neutral was first):
# CLASS_INDICES = {0: 'neutral', 1: 'happy'}

# ⚠️ YOU MUST SET THIS CORRECTLY!
# --- 1. Configuration ---
CLASS_INDICES = {
    0: 'angry', 
    1: 'disgust',
    2: 'fear', 
    3: 'happy', 
    4: 'neutral', 
    5: 'sad', 
    6: 'surprise'
}


# --- 2. Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train first!")

model = load_model(MODEL_PATH)
print(f"Model loaded successfully.")

# --- 3. Face Detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Failed to load Haar cascade. Check OpenCV installation.")

# --- 4. Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam.")

print("Press 'q' to quit.")

# --- 5. Real-Time Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract and preprocess face
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, IMAGE_SIZE)
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))  # (1, 48, 48, 1)

        # Predict
        preds = model.predict(roi, verbose=0)[0]
        idx = int(np.argmax(preds))
        emotion = CLASS_INDICES.get(idx, "unknown")
        conf = preds[idx] * 100

        # Display
        label = f"{emotion}: {conf:.1f}%"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Expression Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
cap.release()
cv2.destroyAllWindows()