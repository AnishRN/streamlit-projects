import streamlit as st
import requests

# Title
st.title("🧠 Enhanced Q&A Chatbot (via Hugging Face API)")

# Load Hugging Face token from Streamlit Secrets
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")

if not HF_API_KEY:
    st.error("❌ Hugging Face API token not found! Please add it to your Streamlit secrets.")
    st.stop()

# Model selector
MODEL_OPTIONS = {
    "FLAN-T5 Small (google/flan-t5-small)": "google/flan-t5-small",
    "FLAN-T5 Base (google/flan-t5-base)": "google/flan-t5-base",
    "BLOOM (bigscience/bloom)": "bigscience/bloom",
    "GPT2 (gpt2)": "gpt2",
    "DistilGPT2 (distilgpt2)": "distilgpt2"
}
selected_model_name = st.sidebar.selectbox("🔧 Select Model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_name]

# User input
st.write("💬 Ask any question below:")
user_input = st.text_input("You:")

# Hugging Face inference function
def generate_response(question, model_id):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
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

    # Check if response is successful
    if response.status_code != 200:
        return f"⚠️ Error {response.status_code}: {response.text}"

    try:
        result = response.json()
        # Handle both list and dict results
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            return f"⚠️ Unexpected response format: {result}"
    except requests.exceptions.JSONDecodeError:
        return "❌ Failed to decode response from Hugging Face API."

# Display response
if user_input:
    with st.spinner("Generating answer..."):
        answer = generate_response(user_input, selected_model)
        st.success(answer)
else:
    st.info("Enter a question to begin.")
