import streamlit as st

# Set page config first!
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Then import other libraries and write the rest of your code.
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_best.keras")

model = load_model()

# Define class labels (same order as during training)
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to preprocess the image
def preprocess_image(image):
    try:
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((224, 224))  # Resize
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

st.title("Brain Tumor Classification")
st.write("Upload an MRI scan to classify the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    if processed_image is not None:
        with st.spinner("Classifying..."):
            # Make prediction
            prediction = model.predict(processed_image)

            # Debugging: Print prediction details
            st.write("Prediction vector:", prediction)
            st.write("Prediction shape:", prediction.shape)

            predicted_class_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_index]

        # Format and display confidence scores, etc.
        confidence_scores = {class_labels[i]: f"{prediction[0][i] * 100:.2f}%" for i in range(len(class_labels))}
        st.success(f"Predicted Tumor Type: `{predicted_class}`")
        st.write("Confidence Scores:")
        for tumor_type, confidence in confidence_scores.items():
            st.write(f"- **{tumor_type}:** {confidence}")
