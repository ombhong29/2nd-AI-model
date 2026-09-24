
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the same CNN architecture as in training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The input features for the linear layer depend on the image size
            # Our images were 250x250, so after two MaxPool2d layers (stride 2),
            # the size becomes 250 / 4 = 62.5, which is floored to 62.
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Configuration
IMAGE_SIZE = (250, 250)
CLASS_NAMES = ['keyboard', 'mouse'] # Make sure this matches your dataset classes
MODEL_PATH = 'simple_cnn_model.pth' # Path to your saved model

# Load the trained model
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transformations (must match training transformations)
def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Predict function
def predict(image, model, transform):
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)

    return CLASS_NAMES[predicted_idx.item()], probabilities[0][predicted_idx.item()].item()

# Streamlit app layout
st.title("Image Classifier: Keyboard vs. Mouse")
st.write("Upload an image and let the model predict if it's a keyboard or a mouse.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load model and transforms
    model = load_model()
    transform = get_transforms()

    # Make prediction
    predicted_class, confidence = predict(image, model, transform)

    st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

# How to run the app:
st.markdown("""
To run this Streamlit app locally, save the above content as `app.py` in the same directory as `simple_cnn_model.pth` and `requirements.txt`.
Then, open your terminal in that directory and run:
```bash
streamlit run app.py
```
"""
)
