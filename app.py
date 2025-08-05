import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import os

st.title("🪖 Helmet Detection with YOLOv5")
st.write("Upload an image to detect whether helmets are worn or not.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Load custom-trained YOLOv5 model (make sure best.pt is in the same directory or provide full path)
        model_path = "best.pt"  # Update this if your file is in a different path
        if not os.path.exists(model_path):
            st.error("Model file not found. Please ensure 'best.pt' is in the app directory.")
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

            # Inference
            results = model(tmp.name)

            # Display detection results
            st.image(np.squeeze(results.render()), caption="Helmet Detection Results", use_column_width=True)

            # Optional: Print raw detections
            st.write("Detection Details:")
            st.write(results.pandas().xyxy[0])
