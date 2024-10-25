import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('trained_model(1).h5')

# Define function to preprocess image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to 64x64 as required by the model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define function to make predictions
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return class_idx, confidence

# Streamlit app interface
st.title("Rock Image Classification")
st.write("Upload an image to classify the rock type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    if st.button("Classify"):
        st.write("Classifying...")
        class_idx, confidence = predict_image(image)

        # Create a popup message showing the result
        st.success(f"Predicted Class: {class_idx}")
        st.info(f"Confidence: {confidence:.2f}")
