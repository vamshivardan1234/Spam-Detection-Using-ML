﻿# Spam-Detection-Using-ML
# 📩 SMS Spam Detector

A simple machine learning web app that classifies SMS messages as Spam or Not Spam using Natural Language Processing (NLP) and a Naive Bayes classifier.

## 🚀 Features
- Text preprocessing using NLTK
- Bag-of-Words model using CountVectorizer
- Trained using Multinomial Naive Bayes
- Web app built with Streamlit

## 🧠 Dataset
- Source: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- ~5,500 labeled SMS messages (ham/spam)

## 🛠️ How to Run

```bash
pip install pandas scikit-learn nltk streamlit
streamlit run sms_spam_detector.py
