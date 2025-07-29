import whisper
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Load the smallest model (faster for testing)
model = whisper.load_model("tiny")
audio_path = "C:/Users/Noman/Desktop/Evolvian/truescan-ai/fake_audio/fake_2.mp3"
# Test transcription (we'll extend this to deepfake detection later)
result = model.transcribe(audio_path)  # You’ll need a sample .wav file
print(result["text"])

audio =  whisper.audio.load_audio(audio_path)
# Compute the mel spectrogram for the loaded audio
mel_spectrogram = whisper.log_mel_spectrogram(audio).cpu().numpy()

print("Feature shape:", mel_spectrogram.shape)  # For 30s audio, usually (80, ~3000)

# Save mel spectrogram for later analysis
np.save("audio_features.npy", mel_spectrogram)