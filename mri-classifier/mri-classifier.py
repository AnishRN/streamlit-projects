import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- Load the model ---
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "mri-classifier.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# --- Class labels and links ---
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'No Tumor']
info_links = {
    'Glioma Tumor': 'https://www.cancer.gov/types/brain/patient/adult-glioma-treatment-pdq',
    'Meningioma Tumor': 'https://www.mayoclinic.org/diseases-conditions/meningioma',
    'Pituitary Tumor': 'https://www.hopkinsmedicine.org/health/conditions-and-diseases/pituitary-tumors',
    'No Tumor': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2723141/'
}

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI brain scan to predict the type of tumor (if any).")

st.markdown("""
### 🧾 About this App
This web application is built using a Convolutional Neural Network (CNN) trained to classify brain MRI images into the following categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

💡 **Note:** Ensure the uploaded MRI image is grayscale or converted from RGB, properly centered, and shows clear brain regions.

This tool is for educational and demonstration purposes only. For medical advice, consult a professional.
""")

uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((128, 128))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))

    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)
    st.write("")

    prediction = model.predict(image_array)
    pred_class_index = np.argmax(prediction)
    pred_class = class_labels[pred_class_index]
    confidence = float(np.max(prediction)) * 100

    st.success(f"🧠 **Predicted Condition:** {pred_class}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}%")

    st.markdown(f"[📚 Learn more about {pred_class}]({info_links[pred_class]})", unsafe_allow_html=True)

    with st.expander("🔬 Show all class probabilities"):
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {prediction[0][i]*100:.2f}%")
