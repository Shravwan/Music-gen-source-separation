import os
import torch
import torchaudio
import soundfile as sf
from main1.model_violin import ViolinClassifier
from torchvision import transforms

# Constants
ROOT_DIR = r"D:\WSAI\data1\split\by_raga"
MODEL_PATH = r"D:\WSAI\main1\violin_model.pth"
SAMPLE_RATE = 22050
CHUNK_DURATION = 2.0  # seconds
THRESHOLD = 0.5

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViolinClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# MelSpectrogram transform
mel_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(SAMPLE_RATE),
    torchaudio.transforms.AmplitudeToDB()
)

# Pad/crop to fixed size
transform = transforms.Lambda(lambda x: x[:, :128, :128])

def extract_violin(audio_path, output_path):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    chunk_size = int(CHUNK_DURATION * SAMPLE_RATE)
    chunks = waveform.squeeze(0).unfold(0, chunk_size, chunk_size)
    
    violin_segments = []
    for i in range(chunks.size(0)):
        chunk = chunks[i].unsqueeze(0)
        mel = mel_transform(chunk)
        mel = transform(mel)
        mel = mel.to(device)
        with torch.no_grad():
            pred = model(mel.unsqueeze(0))
            if pred.item() > THRESHOLD:
                violin_segments.append(chunk.cpu().numpy())

    if violin_segments:
        output_audio = torch.cat([torch.tensor(seg) for seg in violin_segments], dim=-1)
        sf.write(output_path, output_audio.squeeze().numpy(), SAMPLE_RATE)
        print(f"Violin extracted: {output_path}")
    else:
        print(f"No violin detected in: {audio_path}")

# Main loop
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith("_other.wav"):
            other_path = os.path.join(root, file)
            violin_path = other_path.replace("_other.wav", "_violin.wav")

            if os.path.exists(violin_path):
                print(f"✅ Skipping {violin_path}, already exists.")
                continue

            print(f"🔍 Processing {other_path}...")
            extract_violin(other_path, violin_path)
