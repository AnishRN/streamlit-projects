import streamlit as st
import requests
import os

# Title
st.title("🧠 Enhanced Q&A Chatbot (via Hugging Face API)")

# Load Hugging Face token from Streamlit Secrets
HF_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN", "")

if not HF_API_TOKEN:
    st.error("Hugging Face API token not found! Please add it to your Streamlit secrets.")
    st.stop()

# Model selector
MODEL_OPTIONS = {
    "FLAN-T5 Small (google/flan-t5-small)": "google/flan-t5-small",
    "FLAN-T5 Base (google/flan-t5-base)": "google/flan-t5-base",
    "BLOOM (bigscience/bloom)": "bigscience/bloom",
    "Falcon-7B Instruct (tiiuae/falcon-7b-instruct)": "tiiuae/falcon-7b-instruct",
    "Mistral-7B (mistralai/Mistral-7B-Instruct-v0.1)": "mistralai/Mistral-7B-Instruct-v0.1"
}

selected_model_name = st.sidebar.selectbox("🔧 Select Model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_name]

# User input
st.write("💬 Ask any question below:")
user_input = st.text_input("You:")

# Hugging Face inference function
def generate_response(question, model_id):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {
        "inputs": f"Question: {question} Answer:",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7
        },
        "options": {
            "wait_for_model": True
        }
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_id}",
        headers=headers,
        json=payload
    )
    result = response.json()
    if isinstance(result, list):
        return result[0]["generated_text"]
    elif "generated_text" in result:
        return result["generated_text"]
    else:
        return f"⚠️ Error: {result.get('error', 'Unknown error')}"

# Display response
if user_input:
    with st.spinner("Generating answer..."):
        answer = generate_response(user_input, selected_model)
        st.success(answer)
else:
    st.info("Enter a question to begin.")
