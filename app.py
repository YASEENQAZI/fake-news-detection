
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pre-load NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("Fake News Detection App")
st.write("Paste any news article below to check if it is **Fake** or **Real**.")

user_input = st.text_area("Enter News Article Text")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        clean_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]
        if prediction == 1:
            st.success("This article looks **Real News** 📰")
        else:
            st.error("This article looks **Fake News** 🚨")
