import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set up the Streamlit page
st.set_page_config(page_title="Sign Language Translator", layout="centered")

# Load your CNN model
@st.cache_resource
def load_cnn_model():
    try:
        # Update this path to your model.h5 location
        model_path = "C:\Users\ajite\Minor_Project\Sign_Language_Translator\model.h5"
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load class names (replace with your actual class names)
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
               'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = load_cnn_model()

st.title("🧠 Real-Time Sign Language Translator")
st.markdown("Using your trained CNN model with OpenCV + Streamlit")

# Add a sidebar for additional controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)

def preprocess_frame(frame):
    """Preprocess frame for CNN prediction"""
    # Resize and preprocess for your specific model
    frame = cv2.resize(frame, (128, 128))
    frame = img_to_array(frame)
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_sign(frame):
    """Make prediction on a single frame"""
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)[0]
    max_idx = np.argmax(predictions)
    confidence = predictions[max_idx]
    predicted_class = CLASS_NAMES[max_idx]
    return predicted_class, confidence

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
result_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        st.stop()

    try:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not detected")
                break
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Get prediction
            predicted_class, confidence = predict_sign(frame)
            
            # Only show prediction if confidence is above threshold
            if confidence > confidence_threshold:
                # Add prediction to frame
                label = f"{predicted_class} ({confidence:.2f})"
                cv2.putText(display_frame, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show prediction separately
                result_placeholder.success(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                result_placeholder.warning("No confident prediction")
            
            # Display the frame
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(display_frame)

    finally:
        cap.release()