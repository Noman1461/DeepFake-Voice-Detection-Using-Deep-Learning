import whisper
import torch
import numpy as np
from pathlib import Path  # Better than os.path for file handling
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class AudioFeatureExtractor:
    def __init__(self, model_size="tiny"):
        self.model = whisper.load_model(model_size)
    
    def get_features(self, audio_path):
        """Returns standardized features for one audio file"""
        audio = whisper.load_audio(audio_path)
        mel = whisper.log_mel_spectrogram(audio)
        # Convert to numpy (more compatible with ML libraries)
        mel_np = mel.numpy()
        length = mel_np.shape[1]

        return mel_np, length  # Shape: (80, variable_length)

def collate_fn(batch):
    """Pad variable-length features to create equal-length batches"""
    mels = [item['mel'] for item in batch]
    lengths = [item['length'] for item in batch]
    labels =[item['label'] for item in batch]
    #lengths = [length for (_, length) in batch]
    max_length = max(lengths)
    print("max length is:", max_length)
    padded_batch = np.stack([
        np.pad(mel, ((0,0), (0, max_length - mel.shape[1])), 
               mode='constant') 
        for mel in mels
    ])
    # Convert back to tensor
    batch_tensor = torch.from_numpy(padded_batch).float()
    lengths_tensor = torch.tensor(lengths)
    labels_tensor = torch.tensor(labels).float()

    return batch_tensor,lengths_tensor, labels_tensor  

class AudioDataset(Dataset):
    def __init__(self, real_dir="C:/Users/Noman/Desktop/Evolvian/truescan-ai/real_audio",fake_dir="C:/Users/Noman/Desktop/Evolvian/truescan-ai/fake_audio", model_size="tiny"):
        self.extractor = AudioFeatureExtractor(model_size)
        
        #get all file paths with labels
        self.samples = []

        #real audio (label 0)
        real_files = list(Path(real_dir).glob("*.[wm][ap][v3]"))
        print(f"Number of real files: {len(real_files)}")

        if not real_files:
            raise ValueError(f"No audio files in {real_dir}")

        self.samples.extend([(str(f),0) for f in real_files])

        #fake audio (label 1)
        fake_files = list(Path(fake_dir).glob("*.[wm][ap][v3]"))
        print(f"Number of fake files: {len(fake_files)}")

        if not fake_files:
            raise ValueError(f"No audio files in {fake_dir}")
        
        self.samples.extend([(str(f),1) for f in fake_files])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        mel, length = self.extractor.get_features(path)
        return {
            'mel': mel,
            'length': length,
            'label': label,
            'path': path  # Useful for debugging
        }
    
def plot_spectrogram(mel, title=""):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simple test
    dataset = AudioDataset()
    print(f"Loaded {len(dataset)} samples")

    real_sample = next(item for item in dataset if item['label'] == 0)
    fake_sample = next(item for item in dataset if item['label'] == 1)
    
    plot_spectrogram(real_sample['mel'], "Real Voice (Noman)")
    plot_spectrogram(fake_sample['mel'], "AI-Generated Voice (Eleven Labs)")

