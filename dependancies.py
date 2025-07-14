'''
# Let's check the versions of key libraries that would be needed for this task
import sys
print(f"Python version: {sys.version}")

# Try to import libraries we'll need, and install if not available
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not installed, will need to install")

try:
    import torchaudio
    print(f"TorchAudio version: {torchaudio.__version__}")
except ImportError:
    print("TorchAudio not installed, will need to install")

try:
    import librosa
    print(f"Librosa version: {librosa.__version__}")
except ImportError:
    print("Librosa not installed, will need to install")

try:
    import demucs
    print(f"Demucs version: {demucs.__version__ if hasattr(demucs, '__version__') else 'Unknown'}")
except ImportError:
    print("Demucs not installed, will need to install")

try:
    import spleeter
    print(f"Spleeter version: {spleeter.__version__ if hasattr(spleeter, '__version__') else 'Unknown'}")
except ImportError:
    print("Spleeter not installed, will need to install")'''
import shutil
import os
shutil.rmtree(os.path.expanduser("~/.cache/torch/pyannote"), ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb"), ignore_errors=True)
