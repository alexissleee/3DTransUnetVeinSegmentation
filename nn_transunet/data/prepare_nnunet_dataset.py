import os
import numpy as np
import cv2
import json
from glob import glob

# ---------------- CONFIG ----------------
raw_data_folder = "data"  # your original folder containing imgs/ and masks/
nnunet_task_folder = "nnUNet_raw_data_base/Task001_Dataset001_MyData"

# Class mapping
# 0 = background, 1 = boundary, 2 = center
num_classes = 3
class_names = {0: "background", 1: "boundary", 2: "center"}
modality = {0: "CT"}  # name your modality (CT/MR/etc.)

# ---------------- SETUP FOLDERS ----------------
imagesTr_folder = os.path.join(nnunet_task_folder, "imagesTr")
labelsTr_folder = os.path.join(nnunet_task_folder, "labelsTr")
os.makedirs(imagesTr_folder, exist_ok=True)
os.makedirs(labelsTr_folder, exist_ok=True)

# ---------------- CONVERT AND SAVE ----------------
img_files = sorted(glob(os.path.join(raw_data_folder, "imgs", "*.bmp")))
mask_files = sorted(glob(os.path.join(raw_data_folder, "masks", "*.png")))

assert len(img_files) == len(mask_files), "Mismatch between images and masks!"

training_list = []

for idx, (img_path, mask_path) in enumerate(zip(img_files, mask_files)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    # Normalize image to 0-1 (optional)
    img = img / 255.0

    # Build filenames
    base_name = f"{idx:04d}"
    img_file = os.path.join(imagesTr_folder, f"{base_name}.npz")
    mask_file = os.path.join(labelsTr_folder, f"{base_name}_mask.npz")

    # Save separately
    np.savez_compressed(img_file, data=img[np.newaxis, ...])   # add channel dim (C,H,W)
    np.savez_compressed(mask_file, data=mask[np.newaxis, ...])

    # Record for dataset.json
    training_list.append({
        "image": f"imagesTr/{base_name}.npz",
        "label": f"labelsTr/{base_name}_mask.npz"
    })

print(f"✅ Converted {len(img_files)} slices to nnU-Net format.")

# ---------------- CREATE dataset.json ----------------
dataset_json = {
    "name": "Dataset001_MyData",
    "description": "My 2D dataset",
    "tensorImageSize": "2D",
    "reference": "",
    "licence": "",
    "release": "0.0",
    "modality": modality,
    "labels": class_names,
    "numTraining": len(training_list),
    "numTest": 0,
    "training": training_list,
    "test": []
}

with open(os.path.join(nnunet_task_folder, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json created at {nnunet_task_folder}")
