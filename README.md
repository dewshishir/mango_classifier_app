# Bangladeshi Mango Variety Classifier

A simple Streamlit web application that classifies Bangladeshi mango varieties from images using a pre-trained deep learning model.

## Features

- Upload a mango image (JPG, JPEG, or PNG)
- Get a predicted mango variety with confidence score
- View the top-3 predicted varieties with probabilities
- Supports ResNet18, ResNet50, and EfficientNetB0 model architectures

## Project Structure

```
.
├── app.py              # Streamlit application entry point
├── best_model.pt       # Trained PyTorch model weights
├── metadata.pkl        # Model metadata (class names, transforms, model name)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone or download this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app from the project directory:

```bash
streamlit run app.py
```

Then open your browser to the URL shown in the terminal (usually `http://localhost:8501`).

Upload an image of a mango and the app will display the predicted variety along with the confidence score.

## Model Details

- **Framework:** PyTorch + Torchvision
- **Supported architectures:** ResNet18, ResNet50, EfficientNetB0
- **Input:** RGB mango images
- **Output:** Predicted Bangladeshi mango variety and confidence percentage
- Model configuration, class names, and image preprocessing transforms are loaded from `metadata.pkl`.

## Dependencies

See `requirements.txt` for the full list. Key packages include:

- streamlit
- torch
- torchvision
- Pillow
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn

## License

This project is provided as-is for educational and demonstration purposes.
