import streamlit as st
# import pip
# pip.main(['install','tensorflow'])
import tensorflow as tf
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np

# Use the DRIVE_URL from st.secrets
# url = st.secrets['DRIVE_URL']  # The URL is now fetched from secrets
# Modify the URL for direct download from Google Drive
file_id = '1MDkq7qxw4Kj7_ResPR9XcgyRNrFjRLYU'  # Replace with your actual file ID
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Send a request to the URL and download the model
response = requests.get(url)

# Save the downloaded content as a .h5 file
model_path = 'classifier_model.h5'

import os

model_path = 'classifier_model.h5'
if not os.path.exists(model_path):
    print(f"Model file does not exist at path: {model_path}")
else:
    print(f"Model file exists at path: {model_path}")

with open(model_path, 'wb') as file:
    file.write(response.content)

# Load the model into TensorFlow
model = tf.keras.models.load_model(model_path)

# Define a function for image classification
def classify_image(image):
    # Preprocess the image (resize and normalize)
    image = image.resize((32, 32))  # Resize to match the input size of the model
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)  # Get the index of the highest prediction
    class_names = ['Class 1', 'Class 2', 'Class 3']  # Replace with your actual class names
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