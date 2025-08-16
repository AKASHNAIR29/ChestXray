# ======================= Chest X-ray Diagnosis System using Streamlit and TensorFlow =======================

# Import required libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# ======================= Model Loading =======================

# Load the pre-trained model from the 'model' directory
model_path = os.path.join(os.path.dirname(__file__), "model/xray_cnn_model.keras")
model = tf.keras.models.load_model(model_path)

# Define class labels (output classes)
class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']


# ======================= Preprocessing =======================

# Function to preprocess the image before feeding it into the model
def preprocess_image(img):
    # Resize image to match the model input size (100x100)
    img = img.resize((100, 100))
    # Normalize pixel values to range [0, 1]
    img = np.array(img) / 255.0
    # Add an extra batch dimension to the image array
    # This is necessary because the model expects a batch of images, even if it's just one image
    img = img[np.newaxis, ...]
    return img


# ======================= Streamlit UI =======================

# Set up Streamlit page configuration
st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🩺", layout="centered")

# Display title and description using custom HTML formatting
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Chest X-ray Diagnosis System</h1>
    <p style='text-align: center; font-size:18px;'>
        Upload a chest X-ray image to automatically detect signs of <b>COVID-19</b>, <b>Pneumonia</b>, or a <b>Normal</b> scan using a trained deep learning model.
    </p>
""", unsafe_allow_html=True)

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image and convert it to RGB format
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Display a spinner while the model makes a prediction
    with st.spinner("🧠 Analyzing the X-ray..."):
        # Preprocess the uploaded image
        processed_image = preprocess_image(image)
        # Make the prediction using the pre-trained model
        prediction = model.predict(processed_image)[0]
        # Get the predicted class based on the highest probability
        predicted_class = class_names[np.argmax(prediction)]

    # Display the predicted class
    st.success(f"### 🩺 Prediction: `{predicted_class}`")

    # Display prediction probabilities in a bar chart
    st.subheader("📊 Prediction Probabilities")
    prob_display = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    st.bar_chart(prob_display)

    # Display full prediction probabilities in JSON format
    with st.expander("See full confidence scores"):
        st.json(prob_display)


# ======================= Footer =======================

st.markdown("""
    <hr style="border:0.5px solid #ddd;">
    <div style="text-align:center; font-size: 14px;">
        Developed by Akash Nair & Jagruti Patil | UCD MSc Project<br>
        Model Accuracy: <b>~95%</b> | Powered by TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)