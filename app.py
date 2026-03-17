import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="Mango Classifier", page_icon="🥭", layout="centered")

# Title
st.title("🥭 Bangladeshi Mango Variety Classifier")
st.write("Upload an image of a mango and the model will predict its variety.")

# Function to load model and metadata (cached)
@st.cache_resource
def load_model_and_metadata():
    # Load metadata
    metadata = joblib.load('metadata.pkl')
    class_names = metadata['class_names']
    model_name = metadata['model_name']
    transform = metadata['transform']  # val_transform (resize + normalize)

    # Build model architecture (weights=None instead of pretrained=False)
    if model_name == 'ResNet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.to(device)
    model.eval()

    return model, class_names, transform, device

# Load model
try:
    model, class_names, transform, device = load_model_and_metadata()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Choose a mango image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = class_names[predicted.item()]
    confidence_percent = confidence.item() * 100

    # Show results
    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Predicted Variety:** 🥭 **{class_name}**")
    st.write(f"**Confidence:** {confidence_percent:.2f}%")

    # Progress bar
    st.progress(int(confidence_percent))

    # Top-3 predictions (optional)
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    st.write("**Top 3 Probabilities:**")
    for i in range(3):
        st.write(f"- {class_names[top3_idx[0][i]]}: {top3_prob[0][i]*100:.2f}%")