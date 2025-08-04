import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile

st.title("📊 YOLOv5 vs R-CNN Comparison")
st.write("Upload an image to see detection results from both models.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Load YOLOv5 (replace with your path or training weights)
        yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        yolov5_results = yolov5(tmp.name)

        # Render YOLOv5 results
        st.image(np.squeeze(yolov5_results.render()), caption="YOLOv5 Detection", use_column_width=True)

        # If you have an R-CNN model, load and predict here
        st.info("⚠️ R-CNN prediction placeholder — add your R-CNN code here")
