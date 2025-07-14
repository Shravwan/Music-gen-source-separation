import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from main1.model_flute import FluteClassifier # Make sure model.py is in the same folder

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(filepath)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(sr)(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, torch.tensor(label, dtype=torch.float32)

# Transform to fix input size (1, 128, 128)
transform = transforms.Lambda(lambda x: x[:, :128, :128])

# File loading utility
def get_filepaths(data_dir):
    filepaths = []
    labels = []
    for label, subdir in enumerate(['nonflute', 'flute']):
        folder = os.path.join(data_dir, subdir)
        for file in os.listdir(folder):
            if file.endswith(('.wav', '.mp3', '.flac')):
                filepaths.append(os.path.join(folder, file))
                labels.append(label)
    return filepaths, labels

# Training function
def train(data_dir, model_out='flute_cnn.pth', epochs=10, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filepaths, labels = get_filepaths(data_dir)
    train_files, val_files, train_labels, val_labels = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels
    )

    train_dataset = AudioDataset(train_files, train_labels, transform)
    val_dataset = AudioDataset(val_files, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = FluteClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.unsqueeze(1).to(device)  # Shape: (B, 1, H, W), (B, 1)
            #x = x.squeeze(1)  # Remove channel dim: (B, 1, 128, 128) → (B, 128, 128)
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_out)
    print(f"Model saved to {model_out}")

# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Path to training folder with 'violin' and 'nonviolin'")
    parser.add_argument('--out', default='violin_cnn.pth', help="Output .pth file for model")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    train(args.data, args.out, args.epochs, args.batch)
