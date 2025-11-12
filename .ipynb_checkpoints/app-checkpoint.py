import streamlit as st
import pickle
import re
import nltk

# Load ML models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))  # <-- only if you saved LabelEncoder

# Function to clean text
def clean_text(text):
    clean_txt = re.sub(r'http\S+', '', text)
    clean_txt = re.sub(r'@[A-Za-z0-9]+', '', clean_txt)
    clean_txt = re.sub(r'#', '', clean_txt)
    clean_txt = re.sub(r'[^A-Za-z0-9\s]+', '', clean_txt)
    clean_txt = re.sub(r'RT|cc|CC|rt', '', clean_txt)
    clean_txt = re.sub(r'\s+', ' ', clean_txt)
    return clean_txt.strip()

# Streamlit UI
st.title("CV Parser")
uploaded_file = st.file_uploader("Upload Resume:", type=["txt", "pdf"])

if uploaded_file:
    try:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode("utf-8")
    except Exception:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode("latin-1")

    resume_text = clean_text(resume_text)

    # Vectorize and predict
    vectorized_resume = tfidf.transform([resume_text])
    prediction = clf.predict(vectorized_resume)[0]

    # Map prediction to label
    category_mapping = dict(enumerate(le.classes_))
    result = category_mapping.get(prediction, "Unknown")

    st.success(f"Predicted Job Category: **{result}**")
