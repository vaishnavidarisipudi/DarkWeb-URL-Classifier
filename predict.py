import re
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading Trained Model...")

model = load_model("models/textcnn_model.keras")

print("Model Loaded Successfully!")

print("\nLoading Tokenizer...")

with open("output/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer Loaded Successfully!")

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove only http://, https:// and www.
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"^www\.", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    return text
print("\n====================================")
print(" Dark Web URL Detector")
print("====================================")

url = input("\nEnter URL: ")

# Clean the entered URL
clean_url = clean_text(url)

print("\nCleaned URL:")
print(clean_url)

# Convert URL into sequence
sequence = tokenizer.texts_to_sequences([clean_url])

print("\nSequence:")
print(sequence)

# Pad sequence
padded_sequence = pad_sequences(
    sequence,
    maxlen=100,
    padding="post"
)

print("\nPadded Sequence:")
print(padded_sequence)

# Predict probabilities
prediction = model.predict(padded_sequence)

print("\nPrediction Probabilities:")
print(prediction)

# Get predicted class index
predicted_class = np.argmax(prediction)

print("\nPredicted Class Index:")
print(predicted_class)

# Convert class index to class name
labels = [
    "Benign",
    "Defacement",
    "Malware",
    "Phishing"
]

print("\n==============================")
print("Prediction Result")
print("==============================")
print("URL:", url)
print("Prediction:", labels[predicted_class])