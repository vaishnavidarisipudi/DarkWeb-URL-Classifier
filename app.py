import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import re
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Dark Web URL Classifier",
    page_icon="🔐",
    layout="centered"
)

st.title("🔐 Dark Web URL Classifier")

st.write(
    "Enter a URL below to classify it as **Benign, Phishing, Malware, or Defacement**."
)

st.divider()

st.write("### Loading AI Model...")

model = load_model("models/textcnn_model.keras")

with open("output/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.success("✅ Model Loaded Successfully!")

st.divider()

url = st.text_input(
    "🌐 Enter URL",
    placeholder="https://example.com"
)

predict = st.button("🔍 Predict")

def clean_text(text):
    text = text.lower()

    text = re.sub(r"https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    text = re.sub(r"<.*?>", "", text)

    return text

if predict:

    # Check if user entered a URL
    if url == "":
        st.warning("⚠️ Please enter a URL.")

    else:

        # Clean URL
        clean_url = clean_text(url)

        # Convert URL to sequence
        sequence = tokenizer.texts_to_sequences([clean_url])

        # Pad sequence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=100,
            padding="post"
        )

        # Predict
       prediction = model.predict(padded_sequence, verbose=0)

        # Get predicted class
                # Get predicted class
        predicted_class = np.argmax(prediction)

        labels = [
            "Benign",
            "Defacement",
            "Malware",
            "Phishing"
        ]

        st.success(f"Prediction: {labels[predicted_class]}")