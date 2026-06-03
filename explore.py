# explore.py
# This file helps us UNDERSTAND our data before training

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ── Count how many images we have ──────────────────────────────────────────
data_dir = "chest_xray"

for split in ["train", "val", "test"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(data_dir, split, label)
        count = len(os.listdir(path))
        print(f"{split} | {label}: {count} images")

# ── Visualize sample images ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Sample Chest X-Rays", fontsize=16, fontweight='bold')

for row, label in enumerate(["NORMAL", "PNEUMONIA"]):
    folder = os.path.join(data_dir, "train", label)
    images = os.listdir(folder)[:4]  # grab first 4 images
    for col, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = mpimg.imread(img_path)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(label, color='green' if label == 'NORMAL' else 'red')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()
print("✅ Sample image saved as sample_images.png")