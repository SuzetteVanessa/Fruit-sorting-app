import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

# Load your trained model
model = torch.load("fruit_model.pt")
model.eval()

st.title("🍌 Fruit Sorting Web App")
st.write("Upload a fruit image and get its classification.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(image)
    img_tensor = torch.tensor(img_array).permute(2,0,1).unsqueeze(0).float()

    # Run inference
    with torch.no_grad():
        prediction = model(img_tensor)

    # Dummy output handling (replace with your prediction logic)
    predicted_class = torch.argmax(prediction, dim=1).item()
    st.success(f"Prediction: {predicted_class}")
