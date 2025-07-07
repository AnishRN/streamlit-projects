import streamlit as st
import joblib
import re
import string
from nltk.stem import SnowballStemmer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stemmer = SnowballStemmer("english")
def preprocess_and_stem(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|watch\?v=\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)
st.title("🛡️ YouTube Spam Comment Classifier")
st.markdown("Enter a YouTube comment below to predict if it's **spam** or **not spam**.")
user_input = st.text_area("💬 Your Comment:", "")
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment before predicting.")
    else:
        cleaned_text = preprocess_and_stem(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        if prediction == 1:
            st.error("🚨 This comment is likely **SPAM**.")
        else:
            st.success("✅ This comment is likely **NOT SPAM**.")
