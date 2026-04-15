import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import os

# Create folder for alerts
os.makedirs("screenshots", exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detection function
def detect(frame):
    if frame is None:
        return None, "No Frame"

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Add text
    cv2.putText(img, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Save suspicious screenshots
    if len(faces) != 1:
        filename = f"screenshots/alert_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, img)

    return img, f"{status} | Faces detected: {len(faces)}"


# Gradio UI (v6 compatible)
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Smart Exam Proctoring System")
    gr.Markdown("Real-time AI monitoring using webcam")

    webcam = gr.Image(sources=["webcam"], streaming=True, label="Live Camera")
    output_img = gr.Image(label="Processed Output")
    status = gr.Textbox(label="Detection Status")

    # Real-time stream processing
    webcam.stream(
        fn=detect,
        inputs=webcam,
        outputs=[output_img, status]
    )

demo.launch()
