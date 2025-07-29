import torch.nn as nn
import torch.nn.functional as F
import torch
from feature_extractor import DataLoader, AudioDataset, collate_fn
from sklearn.metrics import precision_score, recall_score, f1_score 

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

def train_epoch(model, loader, optimizer, criterion):
    model.train()  # Set model to training mode (important for some layers like dropout, batchnorm)
    total_loss = 0

    for batch in loader:
        mels,length,  labels = batch  # From your collate_fn
        
        optimizer.zero_grad()  # Clear gradients before backward pass 
        outputs = model(mels)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation: compute gradients
        optimizer.step()  # Update weights
        total_loss += loss.item()  # Keep track of loss
    
    return total_loss / len(loader)  # Average loss for the epoch

def evaluate(model, loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            mels,length, labels = batch
            outputs = model(mels)

            # Convert probabilities to binary predictions (0/1)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def compute_metrics(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            mels,length, labels = batch
            outputs = model(mels)
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return precision, recall, f1

if __name__ == "__main__":
    dataset = AudioDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = ImprovedDetector()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        avg_loss = train_epoch(model, loader, optimizer, criterion)
        acc = evaluate(model, loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}")
    
    print("-----------------------------------------------------------------")
    precision, recall, f1 = compute_metrics(model, loader)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    # Save the trained model parameters
    torch.save(model.state_dict(), "detector.pth")
    print("Training complete, model saved as detector.pth")
