import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration Constants
MAX_LEN = 50

st.set_page_config(page_title="Fake Job Posting Predictor", layout="centered")

@st.cache_resource
def load_assets():
    """Load the trained Keras model and tokenizer safely."""
    model = tf.keras.models.load_model("sentiment_lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Load model and tokenizer
with st.spinner("Loading model and tokenizer... Please wait."):
    model, tokenizer = load_assets()

st.title("🚨 Fake Job Posting Detection App")
st.write("Enter the job details below to check whether the listing appears to be **Real** or **Fraudulent**.")

# Input fields corresponding to the concatenated features used during training
col1, col2 = st.columns(2)
with col1:
    job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    location = st.text_input("Location", placeholder="e.g., US, NY, New York")
    department = st.text_input("Department", placeholder="e.g., Engineering")
    salary_range = st.text_input("Salary Range", placeholder="e.g., 50000-70000")
    company_logo = st.selectbox("Has Company Logo?", ["1", "0"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"])

with col2:
    required_experience = st.selectbox("Required Experience", ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable"])
    industry = st.text_input("Industry", placeholder="e.g., Information Technology and Services")

company_profile = st.text_area("Company Profile", placeholder="Brief description of the company...")
description = st.text_area("Job Description", placeholder="Detailed role description...")
requirements = st.text_area("Requirements", placeholder="Required skills and qualifications...")
benefits = st.text_area("Benefits", placeholder="Offered benefits...")

if st.button("Predict Job Status", type="primary"):
    if not job_title.strip() and not description.strip():
        st.warning("Please provide at least a Job Title and Description to run the prediction.")
    else:
        # Concatenate features in the exact same format as training data:
        # df['X'] = title + location + department + salary_range + company_profile + description + requirements + benefits + has_company_logo + employment_type + required_experience + industry
        combined_text = f"{job_title} {location} {department} {salary_range} {company_profile} {description} {requirements} {benefits} {company_logo} {employment_type} {required_experience} {industry}"
        
        # Preprocess input text
        sequence = tokenizer.texts_to_sequences([combined_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        
        # Predict
        prediction = model.predict(padded)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction)) * 100
        
        st.divider()
        
        # Display Result
        if predicted_class == 1:
            st.error(f"⚠️ **Warning: This job posting is predicted to be FRAUDULENT!** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ **This job posting appears to be REAL.** (Confidence: {confidence:.2f}%)")
        
        # Show class probability metrics
        st.subheader("Prediction Probabilities")
        st.metric(label="Fraudulent Probability", value=f"{prediction[0][1]*100:.2f}%")
        st.metric(label="Legitimate Probability", value=f"{prediction[0][0]*100:.2f}%")