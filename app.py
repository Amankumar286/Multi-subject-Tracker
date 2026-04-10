import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Multi-Subject Tracker", layout="wide")
st.title("Target Tracking with YOLOv8")

# Model load karna
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Nano model fast chalta hai deployment pe

model = load_model()

uploaded_file = st.file_system.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Video ko temporary file mein save karna
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Tracking (Jo tere run.py mein tha)
        results = model.track(frame, persist=True)
        
        # Result ko frame pe draw karna
        annotated_frame = results[0].plot()
        
        # Browser mein dikhana
        st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("Tracking Complete!")
