from tensorflow.keras.models import load_model

# Load the original trained model
model = load_model("models/best_textcnn_model.keras", compile=False)

# Save a NEW deployment-compatible model
model.save("models/textcnn_render.keras", include_optimizer=False)

print("New model saved successfully!")