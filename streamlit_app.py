import os
import kaggle
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import streamlit as st

# Download the model from Kaggle (ensure you have Kaggle API credentials setup)
def download_model():
    model_dir = 'meyer_classifier_model'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download the model using the Kaggle API
    try:
        kaggle.api.model_download("isribalaji/meyer_classifier", path=model_dir)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading the model: {e}")

# Define the model path after download
model_path = os.path.join('meyer_classifier_model', 'meyer_classifier_model.h5')

# Download the model from Kaggle if it's not already downloaded
if not os.path.exists(model_path):
    download_model()

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

# Define a function for image classification
def classify_image(image):
    # Preprocess the image (resize and normalize)
    image = image.resize((32, 32))  # Resize to match the input size of the model
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)  # Get the index of the highest prediction
    class_names = model.class_names  # Retrieve class names directly from the model
    predicted_class = class_names[class_idx]
    return predicted_class

# Streamlit interface
st.title('Image Classification App')

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Open and display the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the image
    predicted_class = classify_image(image)

    # Display the result
    st.write(f"Predicted Class: {predicted_class}")