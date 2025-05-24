import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

# CSS to add a blurred, transparent background image
st.markdown(
    """
    <style>
    body {
        background-image: url("https://img.freepik.com/free-photo/cybersecurity-concept-collage-design_23-2151877155.jpg?semt=ais_hybrid&w=740");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .stApp {
        backdrop-filter: blur(6px);
        background-color: rgba(255, 255, 255, 0.2);
        padding: 2rem;
    }

    .stTextArea textarea {
        background-color: rgba(255,255,255,0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize stemmer
ps = PorterStemmer()

# Download stopwords if not already available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)  # Regex tokenization

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
    model = pickle.load(open('model1.pkl', 'rb'))
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure vectorizer1.pkl and model1.pkl exist.")
    st.stop()

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Alert!")
    else:
        st.success("✅ Not Spam")

    st.write("🧹 Processed text:", transformed_sms)
