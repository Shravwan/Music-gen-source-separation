import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as sig


def bandpass_filter(signal, lowcut, highcut, sr, order=6):
    """
    Zero-phase bandpass filter using second-order sections (SOS)
    """
    sos = sig.butter(order, [lowcut, highcut], btype='band', fs=sr, output='sos')
    return sig.sosfiltfilt(sos, signal, axis=-1)


def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def split_others(others_wav_path, output_dir, normalize=True):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Loading audio file: {others_wav_path}")

    # Load stereo audio
    audio, sr = librosa.load(others_wav_path, sr=44100, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # Convert mono to stereo

    print(f"[INFO] Sample rate: {sr}, Channels: {audio.shape[0]}, Duration: {audio.shape[1] / sr:.2f}s")

    # Save the full 'others' audio for reference
    base = os.path.splitext(os.path.basename(others_wav_path))[0]
    sf.write(os.path.join(output_dir, f"{base}_full.wav"), audio.T, sr)

    # Define Carnatic instrument frequency bands (in Hz)
    instrument_bands = {
        "violin": (196, 2630),
        "veena":  (110, 880),
        "flute":  (262, 2093),
        "tanpura": (60, 200),
    }

    for name, (low, high) in instrument_bands.items():
        print(f"[INFO] Processing {name} (band: {low} Hz – {high} Hz)")
        filtered = bandpass_filter(audio, low, high, sr)
        if normalize:
            filtered = normalize_audio(filtered)
        output_path = os.path.join(output_dir, f"{base}_{name}.wav")
        sf.write(output_path, filtered.T, sr)
        print(f"[✓] Saved: {output_path}")

    print("[DONE] Instrument splitting complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:\n  python carnatic_split.py <path_to_others.wav> <output_folder>")
        sys.exit(1)

    others_wav = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(others_wav):
        print(f"[ERROR] File not found: {others_wav}")
        sys.exit(1)

    split_others(others_wav, output_dir)
