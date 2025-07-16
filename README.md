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

## 🧠 Brain Tumor MRI Classifier  
A deep learning model (CNN) trained to classify brain MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.  
🔗 [Try the App](https://app-projects-vbc93rqtnisk9z8whihewb.streamlit.app/)
⚠️ **Note**: This project **cannot be deployed** on Streamlit Community Cloud due to **TensorFlow version restrictions** and **lack of support for large model files**. However, it runs **perfectly on local environments**.

### 🔧 To run locally:
```bash
git clone https://github.com/AnishRN/streamlit-projects.git
cd brain-tumor-classifier/
pip install -r requirements.txt
streamlit run app.py
```
---

## 📁 Projects

- `youtube-spam-checker/`: NLP-based comment classification using Naive Bayes.
- `loan-approval-predictor/`: Structured dataset classification using ensemble models.
- `brain-tumor-classifier/`: Deep learning-based image classification using CNN and medical MRI scans.
  
Each project contains:
- `notebook.ipynb`: Complete analysis & model-building notebook
- `.py` script for Streamlit deployment
- `requirements.txt`
- Trained models (`.pkl`,`.h5`)

---

## 🛠 How to Run Locally

```bash
git clone https://github.com/AnishRN/streamlit-projects.git
cd <project-folder-name>
pip install -r requirements.txt
streamlit run <your_script.py>
```

📌 Note
- All the projects and apps mentioned in this repository are ***ONLY FOR EDUCATIONAL PURPOSES!!!***
- All apps are live on Streamlit Cloud (free-tier). Feel free to fork the repo or test them directly online using the links above!

---

✅ You can now:
- Create a `README.md` file in your repo
- Paste the above content
- Commit the file

Let me know if you want a badge, preview GIF, or project-specific sections added!
