import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="Smart Exam Proctor", layout="centered")

st.title("🎥 Smart Exam Proctoring System")

# Create screenshot folder
os.makedirs("screenshots", exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Camera input
img_file_buffer = st.camera_input("Capture Frame")

if img_file_buffer is not None:
    # Convert to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "✅ Normal"
    color = (0, 255, 0)

    # Detection logic
    if len(faces) == 0:
        status = "⚠️ No Face Detected"
        color = (0, 0, 255)
    elif len(faces) > 1:
        status = "🚨 Multiple Faces Detected"
        color = (0, 0, 255)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Add text overlay
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Save screenshot if suspicious
    if len(faces) != 1:
        filename = f"screenshots/alert_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)

    # Show result
    st.image(frame, channels="BGR")

    st.markdown(f"### Status: {status}")
    st.write(f"Faces detected: {len(faces)}")
