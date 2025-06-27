import pandas as pd
import numpy as np
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources with error handling
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")
ps = PorterStemmer()
df = pd.read_csv('spam.csv',encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
corpus = []
for message in df['message']:
    review=re.sub('[^a-zA-Z]', ' ', message).lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))
cv = CountVectorizer(max_features=3000)
x= cv.fit_transform(corpus).toarray()
y= pd.get_dummies(df['label'], drop_first=True).values.ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
model= MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

st.title("📩 SMS Spam Classifier")
user_input = st.text_area("Enter the SMS message")

if st.button("Predict"):
    test_msg = re.sub('[^a-zA-Z]', ' ', user_input).lower().split()
    test_msg = [ps.stem(word) for word in test_msg if word not in stopwords.words('english')]
    final_input = cv.transform([' '.join(test_msg)]).toarray()
    prediction = model.predict(final_input)
    st.write("### 🛑 Spam" if prediction[0] == 1 else "### ✅ Not Spam")

# Evaluation
st.sidebar.subheader("Model Performance")
st.sidebar.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
st.sidebar.text("Classification Report:")
st.sidebar.text(classification_report(y_test, y_pred))
