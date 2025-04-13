![header](https://raw.githubusercontent.com/AbidHasanRafi/brain-tumor-detection-streamlit/main/header.png) 

# Brain Tumor MRI Classification using CNN

This project is a lightweight, interactive Streamlit application for classifying brain MRI scans into **Glioma**, **Meningioma**, or **Pituitary** tumor categories. It uses a custom-trained Convolutional Neural Network (CNN) to analyze grayscale MRI images and predict tumor types with associated confidence scores.

---

## Features

- Upload a `.pth` trained PyTorch model dynamically
- Upload any brain MRI scan image (`.jpg`, `.jpeg`, `.png`)
- Preprocesses image: grayscale, resize (256x256), normalize
- Displays prediction probabilities and bar chart visualization
- Clean UI powered by **Streamlit**
- Completely offline inference (no external API needed)

---

## Dataset

We used the popular **Brain Tumor Classification (MRI)** dataset from Kaggle:

📂 **Dataset Link:**  
🔗 [https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/data](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/data)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/brain-tumor-classifier.git
   cd brain-tumor-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## Model Training (Coming Soon)

> You can use this section to add your model training script.

### Example Training Structure (Lightweight CNN):

```python
# model.py
import torch.nn as nn
import torch.nn.functional as F

class LightBrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
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
```

Once trained, save your model like this:

```python
torch.save(model.state_dict(), "brain_tumor_model.pth")
```

---

## Deployment

To deploy this app online:

```bash
pip install streamlit
streamlit run app.py
```

Or deploy to **[Streamlit Cloud](https://streamlit.io/cloud)** with one click.

---

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- Pillow
- NumPy
- Matplotlib

Add these to a `requirements.txt`:

```txt
streamlit
torch
numpy
pillow
matplotlib
```

---

## Disclaimer

This tool is for **educational and demonstrational** purposes only.  
It is **not** intended for real-life clinical diagnosis.

---

## Author

**Abid Hasan Rafi**  
ECE Student | ML Researcher | Web Programmer  
[Portfolio](https://abid-hasan-rafi.web.app) | [GitHub](https://github.com/AbidHasanRafi) | [LinkedIn](https://linkedin.com/in/abidhasanrafi)

---

## Show some love

If you find this project helpful, feel free to give it a ⭐ on GitHub!