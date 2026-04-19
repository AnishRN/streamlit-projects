import streamlit as st
import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="centered"
)

# ----------------------------
# Load model + vectorizer
# ----------------------------
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("cyberbullying_model_artifacts.pkl")
    return artifacts["model"], artifacts["vectorizer"]

model, vectorizer = load_artifacts()

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ----------------------------
# Preprocessing
# ----------------------------
html_re = re.compile(r"<.*?>")
url_re = re.compile(r"http\S+")
mention_re = re.compile(r"@\S+")
hashtag_re = re.compile(r"#\S+")
alpha_re = re.compile(r"[^a-zA-Z]")


def clean_text(text):
    text = html_re.sub("", text)
    text = url_re.sub("", text)
    text = mention_re.sub("", text)
    text = hashtag_re.sub("", text)
    text = alpha_re.sub(" ", text).lower()

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]

    return " ".join(words)

# ----------------------------
# UI DESIGN
# ----------------------------

st.title("🛡️ Cyberbullying Detection System")
st.markdown("""
### 📌 About this app
This AI model analyzes text and predicts whether it contains **cyberbullying content**.

- Trained on social media tweets  
- Uses NLP + Machine Learning (Logistic Regression)  
- Converts text into numerical features using Bag of Words  

---

### ⚠️ Disclaimer
This tool is for educational purposes only and may not be 100% accurate.
""")

st.divider()

# ----------------------------
# Input Section
# ----------------------------
user_input = st.text_area(
    "✍️ Enter a tweet or message below:",
    placeholder="Type something like: You are useless and nobody likes you..."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Analyze Text"):

    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # preprocess
        cleaned = clean_text(user_input)

        # vectorize
        vector = vectorizer.transform([cleaned])

        # prediction
        prediction = model.predict(vector)[0]

        # probability (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector).max()
        else:
            proba = None

        # ----------------------------
        # Output Section
        # ----------------------------
        st.subheader("📊 Result")

        if "not" in prediction.lower() or prediction == 0:
            st.success("✅ This text is likely NON-CYBERBULLYING")
            st.balloons()
        else:
            st.error("🚨 This text is likely CYBERBULLYING")

        st.write(f"**Predicted Class:** {prediction}")

        if proba:
            st.write(f"**Confidence Score:** {proba:.2f}")

        # ----------------------------
        # Show preprocessing info
        # ----------------------------
        with st.expander("🧹 View Preprocessed Text"):
            st.code(cleaned)

        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. Text cleaning (remove URLs, mentions, symbols)  
            2. Tokenization + stemming  
            3. Bag of Words vectorization  
            4. Logistic Regression prediction  
            """)

# ----------------------------
# Sidebar info
# ----------------------------
st.sidebar.title("📌 Model Info")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Feature Extraction:** Bag of Words")
st.sidebar.write("**Dataset:** Cyberbullying Tweets")
st.sidebar.write("**Output:** Multi-class classification")

st.sidebar.divider()
st.sidebar.write("💡 Tip: Try different toxic and non-toxic sentences!")
