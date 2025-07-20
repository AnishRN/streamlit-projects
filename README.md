# 📊 Streamlit Projects

This repository contains multiple Streamlit web applications built for Machine Learning use cases. Each folder is a separate deployable app.

---

## 🚀 Live Apps

### 1. 🛡️ YouTube Comment Spam Classifier  
Predicts whether a YouTube comment is spam or not using NLP.  
🔗 [Try the App](https://app-projects-codswfznsoai3mlknvqvsa.streamlit.app/)

### 2. 🏦 Loan Approval Predictor  
Predicts whether a loan will be approved based on financial and personal input details using Random Forest and XGBoost models.  
🔗 [Try the App](https://app-projects-bpgjh4riyznxdtbmjen3tc.streamlit.app/)

### 3. 🧠 Brain Tumor MRI Classifier  
A deep learning model (CNN) trained to classify brain MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.  
🔗 [Try the App](https://app-projects-vbc93rqtnisk9z8whihewb.streamlit.app/)  


### 4. 🎭 IMDB Sentiment Analyzer  
A Recurrent Neural Network (RNN) trained on the IMDB movie reviews dataset to classify sentiment as **Positive** or **Negative**.  
🔗 [Try the App](https://app-projects-zr8dr9ccrjqfdtfqm45w9k.streamlit.app/)  


---

## 📁 Projects

- `youtube-spam-checker/`: NLP-based comment classification using Naive Bayes.
- `loan-approval-predictor/`: Structured dataset classification using ensemble models.
- `brain-tumor-classifier/`: Deep learning-based image classification using CNN and medical MRI scans.
- `imdb-sentiment-analyzer/`: Sentiment analysis using RNN (LSTM) on the IMDB reviews dataset.

Each project contains:
- `notebook.ipynb`: Complete analysis & model-building notebook
- `.py` script for Streamlit deployment
- `requirements.txt`
- Trained models (`.pkl`, `.h5`)

---

## 🛠 How to Run Locally

```bash
git clone https://github.com/AnishRN/streamlit-projects.git
cd <project-folder-name>
pip install -r requirements.txt
streamlit run <your_script.py>
```

##⚠️ **Disclaimer**
All the projects in this repository are created for educational, reference, or personal experimentation purposes only.
They are not intended for commercial use, clinical diagnosis, or any field deployment.

Use them responsibly and do not rely on these applications for critical decisions.
