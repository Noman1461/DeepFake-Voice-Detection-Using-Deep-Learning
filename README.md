# Audio Real vs Fake Detection with PyTorch, FastAPI, and Docker

This project implements a deep learning model to classify audio samples as **real** or **fake** using mel spectrogram features, with a complete pipeline from data loading and training to deployment as a REST API. The deployment leverages **FastAPI** for serving the model and supports audio file upload for on-the-fly predictions. Docker is used to containerize the application for easy cross-platform deployment.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Setup and Installation](#setup-and-installation)  
- [Usage](#usage)  
  - [Training the Model](#training-the-model)  
  - [Running the API](#running-the-api)  
  - [Making Predictions](#making-predictions)  
- [Project Structure](#project-structure)  
- [Model Architecture](#model-architecture)  
- [Feature Extraction](#feature-extraction)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This project explores audio classification to distinguish between genuine and fake audio samples. It employs a simple yet effective neural network model built with PyTorch, utilizing mel spectrograms extracted consistently via a wrapped feature extractor class. The trained model is served through a FastAPI web service, which accepts audio file uploads, performs feature extraction, and returns model predictions.

---

## Features

- Custom PyTorch `AudioDataset` with dynamic padding for variable-length mel spectrogram inputs
- Simple neural network classifier with hidden layers and ReLU activation
- Consistent feature extraction pipeline using a reproducible feature extractor (`feature_extractor.py`)
- FastAPI server exposing endpoints for:
  - Health check (`GET /`)
  - Dummy prediction (`GET /predict`)
  - Real audio file upload and inference (`POST /predict-file/`)
- Containerization ready via Docker for consistent deployment environments

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- `pip` package manager
- Docker (optional, for containerized deployment)

### Install Python dependencies

pip install torch torchvision torchaudio fastapi uvicorn python-multipart librosa scikit-learn
_Note:_ `librosa` is only needed if you use it for audio processing instead of your custom extractor.

### (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate # Unix/macOS
.\venv\Scripts\activate # Windows
pip install -r requirements.txt

---

## Usage

### Training the Model

1. Prepare your audio dataset with real and fake samples.
2. Ensure your `AudioDataset` class and `collate_fn` properly load and pad mel spectrograms using `feature_extractor.py`.
3. Run the training script: python train.py

- The training script will:
  - Load data,
  - Train the neural network with binary cross-entropy loss,
  - Save the trained model to `detector.pth`.

### Running the API

Start the FastAPI server locally:
uvicorn api:app --reload

- Access the health check at `http://127.0.0.1:8000/`.
- Access the test dummy prediction at `http://127.0.0.1:8000/predict`.

### Making Predictions

Use the `POST /predict-file/` endpoint to upload an audio file and receive a prediction.

Sample using `curl`: curl -X POST "http://127.0.0.1:8000/predict-file/"
-F "file=@path_to_audio_file.wav"

Response example:
{
"prediction": 0.87,
"filename": "audio_file.wav"
}


---

## Project Structure

├── api.py # FastAPI server with model loading and endpoints

├── train.py # Script for training the model

├── feature_extractor.py # Custom audio feature extraction class (Whisper mel spectrogram)

├── detector.pth # Saved PyTorch model weights

├── Dockerfile # Docker configuration (if applicable)

├── requirements.txt # Python dependencies (optional)

└── README.md # This documentation file


---

## Model Architecture

The current deployed model (`ImprovedDetector`) is a simple feedforward neural network structured as:

- Input: Mel spectrogram features of shape `(batch_size, 80, time_frames)`
- Average pooling over time dimension to shape `(batch_size, 80)`
- Fully connected layer with ReLU activation (hidden layer)
- Output layer with Sigmoid activation to yield probability of real audio (value between 0 and 1)

This provides a solid baseline and allows straightforward extensibility to CNNs or RNNs in future iterations.

---

## Feature Extraction

For consistency, the project uses a custom `AudioFeatureExtractor` class wrapping Whisper’s mel spectrogram extraction. This ensures feature extraction during inference matches that used in training, avoiding discrepancies that cause model degradation.
You can extend or modify `feature_extractor.py` as needed, but keep it consistent throughout the pipeline.

---

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for:

- Model improvements or architectural experimentation
- Robust feature extraction enhancements
- API endpoint improvements or new features
- Dockerization and deployment automation

Please follow standard coding practices and update this README as needed.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Happy audio classifying!**  
If you encounter any issues or have questions, please open an issue or contact the maintainer.
