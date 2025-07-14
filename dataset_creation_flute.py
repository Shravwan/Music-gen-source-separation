import os
import glob
import shutil

# Paths
iras_root = r"D:\WSAI\IRMAS-TrainingData"  # Download and extract IRMAS train folder
output_flute = r"D:\WSAI\main1\dataset\flute"
output_nonflute = r"D:\WSAI\main1\dataset\nonflute"

os.makedirs(output_flute, exist_ok=True)
os.makedirs(output_nonflute, exist_ok=True)

# Copy violin files
for path in glob.glob(os.path.join(iras_root, "flu", "*.wav")):
    shutil.copy(path, output_flute)

# Copy other instrument folders to nonviolin
for folder in os.listdir(iras_root):
    if folder == "flu": continue
    for path in glob.glob(os.path.join(iras_root, folder, "*.wav")):
        shutil.copy(path, output_nonflute)

print("✅ IRMAS dataset prepared:")
print(f"  Flute: {len(os.listdir(output_flute))}")
print(f"  Non‑flute: {len(os.listdir(output_nonflute))}")
