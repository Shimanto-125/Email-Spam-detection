import pandas as pd
import numpy as np
import re
import string

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1','v2']]
    df.columns = ['lebel','message']
    df['lebel_num'] = df.lebel.map({'ham':0, 'spam': 1})
    return df

df = load_data()

#email procesing function
def preprocess_email(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

df['message_clean'] = df['message'].apply(preprocess_email)

#feature extraction

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message_clean'])
y = df['lebel_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

st.title("📧 Email Spam Detection")
st.write("Enter an email message and check if it's **Spam** or **Ham**.")
user_input = st.text_area("Type your email here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email message!")
    else:
        user_clean = preprocess_email(user_input)
        user_vect = tfidf.transform([user_clean])
        prediction = model.predict(user_vect)[0]
        st.success(f"Prediction: {'🚫 Spam' if prediction==1 else '✅ Ham'}")
if st.checkbox("Show Dataset Overview"):
    st.write(df.head())

if st.checkbox("Show Spam vs Ham Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax)
    ax.set_title("Spam vs Ham Emails")
    st.pyplot(fig)

if st.checkbox("Show Confusion Matrix on Test Set"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

