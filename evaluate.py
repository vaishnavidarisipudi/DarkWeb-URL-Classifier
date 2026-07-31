from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
print("Loading Saved Model...")

model = load_model("models/textcnn_model.keras")

print("Model Loaded Successfully!")
print("\nLoading Test Data...")

X_test_pad = np.load("output/X_test_pad.npy")
y_test = np.load("output/y_test.npy")

print("Test Data Loaded Successfully!")
print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test,
    verbose=1
)

y_pred = model.predict(X_test_pad)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nEvaluation Completed!")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")