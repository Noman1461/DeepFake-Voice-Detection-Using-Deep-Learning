from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from feature_extractor import AudioFeatureExtractor
import numpy as np
import io
from pathlib import Path
import tempfile
import os
#uvicorn api:app --reload

extractor = AudioFeatureExtractor(model_size="tiny")  # Adjust model size as needed


class ImprovedDetector(nn.Module):
    def __init__(self, input_size=80, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # First hidden layer
        self.relu = nn.ReLU()                           # Non-linearity
        self.fc2 = nn.Linear(hidden_size, 1)            # Output layer

    def forward(self, x):
        x = x.mean(dim=2)          # Average over time: shape (batch, 80)
        x = self.relu(self.fc1(x)) # Hidden layer with ReLU
        x = torch.sigmoid(self.fc2(x))  # Output probability
        return x.squeeze(1)

app = FastAPI()

model = ImprovedDetector()
model.load_state_dict(torch.load("detector.pth", map_location=torch.device("cpu")))
model.eval()

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}

@app.get("/predict")
def predict_stub():
    # Example: create a fake batch to test
    dummy = torch.zeros((1, 80, 100))  # batch of 1, 80 mel, 100 frames
    with torch.no_grad():
        prob = model(dummy)
    return {"prediction": float(prob.item())}

@app.post("/predict-file/") ##I HAVE TO USE POST METHOD FOR FILE UPLOAD
async def predict_file(file: UploadFile = File(...)):
    # Here you will:
    # - read your uploaded audio file bytes
    # - extract features using your feature_extractor.py
    # - run the model forward
    # - return the prediction

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
    contents = await file.read()
    # if len(contents) > MAX_FILE_SIZE:
    #     return {"error": "File too large. Maximum allowed size is 10MB."}
    # audio_bytes = contents

        # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Use the temporary file path with your extractor
        mel_np, _ = extractor.get_features(tmp_path)

        # Prepare input tensor and model inference ...
        mel_tensor = torch.tensor([mel_np], dtype=torch.float32)
        if mel_tensor.ndim == 2:
            mel_tensor = mel_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(mel_tensor)

        return {"prediction": float(pred.item())}

    finally:
        # Cleanup: remove temp file after processing
        os.remove(tmp_path)
    


