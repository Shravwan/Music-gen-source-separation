import os
os.environ["SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY"] = "copy" 

import torchaudio
import torch
import librosa
import numpy as np
from speechbrain.inference import EncoderClassifier
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import soundfile as sf
import sys

if len(sys.argv) != 3:
    print("Usage: python gender_split.py <input_audio.wav> <output_dir>")
    sys.exit(1)

audio_path = sys.argv[1]
output_dir = sys.argv[2]
base_name = os.path.splitext(os.path.basename(audio_path))[0]

print(f"Processing input audio: {audio_path}")
print(f"Saving gender-separated audio to: {output_dir}")
# Load audio
signal, sr = torchaudio.load(audio_path)
signal = signal.mean(dim=0, keepdim=True)  # mono

# Frame parameters
frame_len = int(2 * sr)  # 2 second segments
hop_len = int(1 * sr)    # 1 second hop
frames = []

# Segment audio into overlapping frames
for start in range(0, signal.shape[1] - frame_len, hop_len):
    segment = signal[:, start:start + frame_len]
    frames.append(segment)

# Load pretrained speaker embedding model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
)

# Get embeddings
embeddings = []
for frame in frames:
    emb = classifier.encode_batch(frame).squeeze().detach().numpy()
    embeddings.append(emb)
embeddings = np.stack(embeddings)

# Cluster speakers (assume 2 speakers – adjust as needed)
kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
labels = kmeans.labels_

# Estimate pitch to decide gender for each cluster
def estimate_pitch(y, sr):
    f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=300)
    return np.nanmean(f0[voiced_flag]) if voiced_flag.any() else 0

cluster_pitches = []
for i in range(2):
    all_segments = [frames[j] for j in range(len(frames)) if labels[j] == i]
    audio_concat = torch.cat(all_segments, dim=1).squeeze().numpy()
    pitch = estimate_pitch(audio_concat, sr)
    cluster_pitches.append(pitch)

# Assign gender (heuristic: lower pitch = male)
male_idx, female_idx = (0, 1) if cluster_pitches[0] < cluster_pitches[1] else (1, 0)

# Save gender-separated audio
male_segments = [frames[i] for i in range(len(frames)) if labels[i] == male_idx]
female_segments = [frames[i] for i in range(len(frames)) if labels[i] == female_idx]

def reconstruct_audio(segments, hop_len, total_len):
    audio_out = torch.zeros(total_len)
    weight = torch.zeros(total_len)

    for i, seg in enumerate(segments):
        start = i * hop_len
        end = start + seg.shape[-1]
        audio_out[start:end] += seg.squeeze()
        weight[start:end] += 1

    # Avoid division by zero
    weight = torch.where(weight == 0, torch.tensor(1.0), weight)
    return audio_out / weight

# Calculate total output length
total_len = signal.shape[1]

# Reconstruct with overlap-add
male_audio = reconstruct_audio(male_segments, hop_len, total_len)
female_audio = reconstruct_audio(female_segments, hop_len, total_len)

# Save
sf.write(os.path.join(output_dir, f"output_male_{base_name}.wav"), male_audio.numpy(), sr)
sf.write(os.path.join(output_dir, f"output_female_{base_name}.wav"), female_audio.numpy(), sr)

