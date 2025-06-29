import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('mnist_digit_recognition.h5')

# Function to preprocess image
def preprocess_image(image):
    image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Draw a digit below and the AI will predict it!")

# Create a canvas for users to draw digits
canvas = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if canvas:
    image = Image.open(canvas)
    st.image(image, caption="Uploaded Digit", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_label}")
