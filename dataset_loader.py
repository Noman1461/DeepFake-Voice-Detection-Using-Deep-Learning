from feature_extractor import AudioFeatureExtractor
from pathlib import Path
import numpy as np

class AudioDataset:
    def __init__(self, real_dir="C:/Users/Noman/Desktop/Evolvian/truescan-ai/real_audio", fake_dir="C:/Users/Noman/Desktop/Evolvian/truescan-ai/fake_audio"):
        self.extractor = AudioFeatureExtractor()
        self.real_files = list(Path(real_dir).glob("*.mp3")) + list(Path(real_dir).glob("*.wav"))
        self.fake_files = list(Path(fake_dir).glob("*.mp3")) + list(Path(fake_dir).glob("*.wav"))
        
    def __len__(self):
        return len(self.real_files) + len(self.fake_files)
    
    def load(self):
        """Loads all audio files with labels (0=real, 1=fake)"""
        features, labels = [], []
        
        # Load real audio
        for file in self.real_files:
            features.append(self.extractor.get_features(str(file)))
            labels.append(0)
            
        # Load fake audio
        for file in self.fake_files:
            features.append(self.extractor.get_features(str(file)))
            labels.append(1)
            
        return features, np.array(labels)

# Test it
if __name__ == "__main__":
    dataset = AudioDataset()
    features, labels = dataset.load()
    print(f"Loaded {len(features)} samples")
    #print(f"First sample shape: {features[0].shape}")
    print(f"Labels: {labels}")