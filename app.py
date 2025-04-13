import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os

# Custom image processing (replaces OpenCV)
class ImageProcessor:
    @staticmethod
    def to_grayscale(image):
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return image
    
    @staticmethod
    def resize(image, size=(256, 256)):
        pil_img = Image.fromarray(image)
        return np.array(pil_img.resize(size, Image.Resampling.LANCZOS))
    
    @staticmethod
    def normalize(image):
        return image / 255.0

# Define the model architecture
class LightBrainTumorCNN(nn.Module):
    def __init__(self):
        super(LightBrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# UI config
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

st.title("🧠 Brain Tumor MRI Classification")
st.write("""
This app classifies brain MRI scans into three categories: 
**Glioma**, **Meningioma**, or **Pituitary** tumor.
""")

# Upload model
model_file = st.file_uploader("Upload your trained model (.pth file)", type=["pth"])
model = None

if model_file is not None:
    try:
        model = LightBrainTumorCNN()
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()
else:
    st.warning("Please upload a `.pth` model file to proceed.")
    st.stop()

# Class names
class_names = ['glioma', 'meningioma', 'pituitary']

# Preprocess function
def preprocess_image(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = ImageProcessor.to_grayscale(image)
        image = ImageProcessor.resize(image)
        image = ImageProcessor.normalize(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return image
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# File uploader for MRI
uploaded_file = st.file_uploader("Upload a brain MRI scan", type=['jpg', 'jpeg', 'png'])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        with st.spinner('Analyzing MRI scan...'):
            input_tensor = preprocess_image(image)
            if input_tensor is not None:
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()

        with col2:
            if input_tensor is not None:
                st.subheader("Prediction Results")

                st.write("**Classification Probabilities:**")
                for i, class_name in enumerate(class_names):
                    prob = probabilities[i].item() * 100
                    st.metric(label=class_name.capitalize(), 
                              value=f"{prob:.2f}%", 
                              delta="HIGHEST" if i == predicted_class else None)

                st.success(f"**Predicted Tumor Type:** {class_names[predicted_class].capitalize()}")

                st.write("**Confidence Distribution:**")
                fig, ax = plt.subplots()
                ax.barh(class_names, probabilities.numpy() * 100, color='skyblue')
                ax.set_xlabel('Confidence (%)')
                ax.set_xlim(0, 100)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.caption("Note: This is a demonstration app. Predictions should not be used for medical diagnosis.")
