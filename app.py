import cv2
import numpy as np
import streamlit as st
try:
    # First try TensorFlow 2.x imports
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    print("Using TensorFlow backend")
except ImportError:
    try:
        # Fallback to standalone Keras
        from keras.models import load_model
        from keras.preprocessing.image import img_to_array
        print("Using standalone Keras")
    except ImportError as e:
        raise ImportError(
            "Failed to import Keras. Please install TensorFlow:\n"
            "pip install tensorflow"
        ) from e
from PIL import Image

# Set up Streamlit
st.set_page_config(page_title="Sign Language Translator", layout="centered")

# Define class names
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Nothing']  # Added 'Nothing' class

@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("model.h5")
        print(f"Model loaded. Input shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def preprocess_frame(frame):
    """Preprocess frame for CNN prediction"""
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to match model
    frame = cv2.resize(frame, (64, 64))
    
    # Convert to array and normalize
    frame = img_to_array(frame)
    frame = frame / 255.0  # For [0,1] range
    # frame = frame / 127.5 - 1.0  # For [-1,1] range
    
    # Add batch dimension
    return np.expand_dims(frame, axis=0)

def predict_sign(frame, model):
    """Make prediction on a single frame"""
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame, verbose=0)[0]
    max_idx = np.argmax(predictions)
    confidence = float(predictions[max_idx])
    
    # Only return prediction if confidence is high enough
    if confidence < 0.7:  # Adjust this threshold as needed
        return "Nothing", 0.0
    return CLASS_NAMES[max_idx], confidence

# Load model
model = load_cnn_model()

# App UI
st.title("🧠 Real-Time Sign Language Translator")
st.markdown("Using your trained CNN model with OpenCV + Streamlit")

# Settings sidebar
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
    show_debug = st.checkbox("Show Debug View")

# Main app
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
result_placeholder = st.empty()
debug_placeholder = st.empty() if show_debug else None

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
            predicted_class, confidence = predict_sign(frame, model)
            
            # Display results
            label = f"{predicted_class} ({confidence:.2f})" if predicted_class != "Nothing" else "No sign detected"
            color = (0, 255, 0) if predicted_class != "Nothing" else (0, 0, 255)
            
            cv2.putText(display_frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Show prediction separately
            if predicted_class != "Nothing":
                result_placeholder.success(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                result_placeholder.warning("No sign detected")
            
            # Display the frame
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(display_frame)
            
            # Debug view
            if show_debug:
                debug_frame = preprocess_frame(frame)[0]
                debug_frame = ((debug_frame - debug_frame.min()) * 255 / (debug_frame.max() - debug_frame.min())).astype('uint8')
                debug_placeholder.image(debug_frame, caption="Preprocessed Input", width=224)

    finally:
        cap.release()