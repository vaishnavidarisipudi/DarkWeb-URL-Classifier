import re
import pandas as pd
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)

print("Libraries loaded successfully!")
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

   # Remove only http://, https:// and www.
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"^www\.", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    return text

sample = "https://www.paypal-login-security.xyz/login.php?id=123"
print("\nOriginal Text:")
print(sample)

print("\nCleaned Text:")
print(clean_text(sample))
print("\nLoading dataset...")

# Read CSV
df = pd.read_csv("dataset/malicious_phish.csv")

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)
print("\nCleaning all URLs...")

df["clean_url"] = df["url"].apply(clean_text)

print("\nFirst 5 Cleaned URLs:")

print(df[["url", "clean_url"]].head())
print("\nEncoding Labels...")

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["type"])

print("\nLabel Mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} --> {i}")

print("\nFirst 5 Rows:")

print(df[["type", "label"]].head())
print("\nNumber of samples in each class:")
print(df["type"].value_counts())
print("\nSplitting Dataset...")

X = df["clean_url"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

url_lengths = X_train.str.len()

print("\nMaximum URL Length:")
print(url_lengths.max())

print("\nAverage URL Length:")
print(url_lengths.mean())

print("\nTraining Data Size:")
print(len(X_train))

print("\nTesting Data Size:")
print(len(X_test))
print("\nCreating Tokenizer...")

# Create tokenizer
tokenizer = Tokenizer(
    char_level=True
)

# Learn all the words from the training data
tokenizer.fit_on_texts(X_train)

print("\nTokenizer Created Successfully!")

print("\nTotal Unique Words:")

print(len(tokenizer.word_index))
print("\nConverting Text into Sequences...")

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

print("\nFirst 5 Training Sequences:")

for i in range(5):
    print(X_train.iloc[i])
    print(X_train_seq[i])
    print()
print("\nPadding Sequences...")

max_length = 100

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length,
    padding="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length,
    padding="post"
)

print("\nShape of Training Data:")
print(X_train_pad.shape)

print("\nShape of Testing Data:")
print(X_test_pad.shape)

print("\nFirst Padded Sequence:")
print(X_train_pad[0])
# Save test data
np.save("output/X_test_pad.npy", X_test_pad)
np.save("output/y_test.npy", y_test)

# Save tokenizer
with open("output/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Test data and tokenizer saved successfully!")


print("\nBuilding TextCNN Model...")

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()

print("Sequential Model Created Successfully!")

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=128
    )
)

print("Embedding Layer Added Successfully!")

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )
)

print("Conv1D Layer Added Successfully!")

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

print("MaxPooling1D Layer Added Successfully!")
model.add(
    GlobalMaxPooling1D()
)

print("GlobalMaxPooling1D Layer Added Successfully!")

model.add(
    Dense(
        128,
        activation="relu"
    )
)

print("Dense Layer Added Successfully!")

model.add(
    Dropout(0.5)
)

print("Dropout Layer Added Successfully!")

model.add(
    Dense(
        4,
        activation="softmax"
    )
)

print("Output Layer Added Successfully!")
print("\nCompiling Model...")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, max_length))

print("Model Compiled Successfully!")

print("\nModel Summary:")
model.summary()

print("\nTraining Model...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_textcnn_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
history = model.fit(
    X_train_pad,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        early_stop,
        checkpoint
    ],
    verbose=1
) 

print("Training Completed Successfully!")

model.save("models/textcnn_model.keras")

print("Model Saved Successfully!")

print("\nPlotting Training Graphs...")

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("output/accuracy_graph.png")
plt.show()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("output/loss_graph.png")
plt.show()

print("Graphs Saved Successfully!")