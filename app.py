import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
import platform
from pathlib import Path

# Set page config early
st.set_page_config(page_title="Sign Language Translator", layout="centered")

# Add YOLOv5 path (alternative if package installation fails)
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
except ImportError:
    st.error("YOLOv5 dependencies not found. Please ensure yolov5 directory is included.")
    st.stop()

@st.cache_resource
def load_model():
    # Force CPU for deployment to avoid CUDA issues
    device = 'cpu'
    try:
        model_path = Path('best.pt')
        if not model_path.exists():
            st.error("Model file 'best.pt' not found!")
            st.stop()
            
        model = DetectMultiBackend(str(model_path), device=device, dnn=False)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, device = load_model()
names = model.names if hasattr(model, 'names') else [''] * model.nc

st.title("🧠 Real-Time Sign Language Translator")
st.markdown("Using your trained YOLOv5 model (`best.pt`) with OpenCV + Streamlit")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        st.stop()

    try:
        while run:  # Use the checkbox state to control the loop
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not detected")
                break

            # Preprocess
            img = letterbox(frame, new_shape=640)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)

            # Inference
            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.float() / 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                pred = model(img_tensor)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

            # Post-process
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
                for *xyxy, conf, cls in pred:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), 
                                 (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

    finally:
        cap.release()