import os
import glob
import shutil

# Paths
iras_root = r"D:\WSAI\IRMAS-TrainingData"  # Download and extract IRMAS train folder
output_violin = r"D:\WSAI\main1\dataset\violin"
output_nonviolin = r"D:\WSAI\main1\dataset\nonviolin"

os.makedirs(output_violin, exist_ok=True)
os.makedirs(output_nonviolin, exist_ok=True)

# Copy violin files
for path in glob.glob(os.path.join(iras_root, "vio", "*.wav")):
    shutil.copy(path, output_violin)

# Copy other instrument folders to nonviolin
for folder in os.listdir(iras_root):
    if folder == "vio": continue
    for path in glob.glob(os.path.join(iras_root, folder, "*.wav")):
        shutil.copy(path, output_nonviolin)

print("✅ IRMAS dataset prepared:")
print(f"  Violin: {len(os.listdir(output_violin))}")
print(f"  Non‑violin: {len(os.listdir(output_nonviolin))}")
