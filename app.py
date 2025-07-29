
import joblib
import re
import streamlit as st

# Custom stopwords
stop_words = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

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
