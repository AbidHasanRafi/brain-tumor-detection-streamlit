import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Custom image processing
class ImageProcessor:
    @staticmethod
    def to_grayscale(image):
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return image

    @staticmethod
    def resize(image, size=(256, 256)):
        pil_img = Image.fromarray(image.astype(np.uint8))
        return np.array(pil_img.resize(size, Image.Resampling.LANCZOS))

    @staticmethod
    def normalize(image):
        return image / 255.0

# Model
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

        self.gradients = None
        self.activations = None

        self.conv3.register_forward_hook(self.activations_hook)
        self.conv3.register_full_backward_hook(self.gradients_hook)

    def activations_hook(self, module, input, output):
        self.activations = output

    def gradients_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations

# Colormap
def create_jet_colormap():
    cdict = {
        'red':   [(0.0, 0.0, 0.0),
                  (0.35, 0.0, 0.0),
                  (0.66, 1.0, 1.0),
                  (0.89, 1.0, 1.0),
                  (1.0, 0.5, 0.5)],
        'green': [(0.0, 0.0, 0.0),
                  (0.125, 0.0, 0.0),
                  (0.375, 1.0, 1.0),
                  (0.64, 1.0, 1.0),
                  (0.91, 0.0, 0.0),
                  (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 0.0, 0.5),
                  (0.11, 1.0, 1.0),
                  (0.34, 1.0, 1.0),
                  (0.65, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]
    }
    return LinearSegmentedColormap('jet', cdict)

# Grad-CAM
def apply_grad_cam(model, input_tensor, target_class=None):
    model.eval()

    # Forward pass
    output = model(input_tensor)

    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()

    # Zero grads & backward pass
    model.zero_grad()
    output[0, target_class].backward()

    # Get gradients and activations
    gradients = model.get_activations_gradient()        # Shape: [1, C, H, W]
    activations = model.get_activations()               # Shape: [1, C, H, W]

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Shape: [C]

    # Weight the channels
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Heatmap computation
    heatmap = torch.mean(activations, dim=1).squeeze()

    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)  # ReLU
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("🧠 Brain Tumor MRI Classification")
st.write("This app classifies brain MRI scans into **Glioma**, **Meningioma**, or **Pituitary** tumor.")

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

class_names = ['glioma', 'meningioma', 'pituitary']

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

uploaded_file = st.file_uploader("Upload a brain MRI scan", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_image = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        with st.spinner('Analyzing MRI scan...'):
            input_tensor = preprocess_image(image)
            if input_tensor is not None:
                input_tensor.requires_grad_()
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()

                # Grad-CAM
                input_tensor.requires_grad = True
                heatmap = apply_grad_cam(model, input_tensor)

                # Resize heatmap
                heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
                heatmap_img = heatmap_img.resize((original_image.shape[1], original_image.shape[0]), Image.Resampling.LANCZOS)
                heatmap = np.array(heatmap_img) / 255.0

                jet = create_jet_colormap()

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                ax1.imshow(original_image, cmap='gray')
                ax1.set_title('Original Image')
                ax1.axis('off')

                ax2.imshow(heatmap, cmap=jet)
                ax2.set_title('Grad-CAM Heatmap')
                ax2.axis('off')

                if len(original_image.shape) == 2:
                    original_image = np.stack([original_image]*3, axis=-1)
                elif original_image.shape[2] == 4:
                    original_image = original_image[:, :, :3]

                heatmap_rgb = plt.cm.jet(heatmap)[..., :3]
                original_image_norm = original_image / 255.0
                superimposed_img = heatmap_rgb * 0.4 + original_image_norm * 0.6
                superimposed_img = np.clip(superimposed_img, 0, 1)

                ax3.imshow(superimposed_img)
                ax3.set_title('Highlighted Regions')
                ax3.axis('off')

                plt.tight_layout()

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
                st.subheader("Grad-CAM Visualization")
                st.write("Heatmap shows regions that most influenced the prediction:")
                st.pyplot(fig)

                st.write("**Confidence Distribution:**")
                fig2, ax = plt.subplots()
                ax.barh(class_names, probabilities.numpy() * 100, color='skyblue')
                ax.set_xlabel('Confidence (%)')
                ax.set_xlim(0, 100)
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.caption("Note: This is a demonstration app. Predictions should not be used for medical diagnosis.")
