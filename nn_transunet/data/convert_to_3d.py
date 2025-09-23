import os
import nibabel as nib
import numpy as np
from PIL import Image

# Root dataset path (contains OA, ICA, ICA2)
dataset_root = "dataset"

# Mapping subfolders to task numbers
tasks = {
    "OA": "001_OA",
    "ICA": "002_ICA",
    "ICA2": "003_ICA2"
}

def convert_image_to_nifti(image_path, output_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    img_array = img_array[..., np.newaxis]  # make (H, W, 1)
    nifti_img = nib.Nifti1Image(img_array, affine=np.eye(4))
    nib.save(nifti_img, output_path)

def convert_mask_to_nifti(mask_path, output_path):
    mask = Image.open(mask_path)
    mask_array = np.array(mask, dtype=np.uint8)
    mask_array = mask_array[..., np.newaxis]
    nifti_mask = nib.Nifti1Image(mask_array, affine=np.eye(4))
    nib.save(nifti_mask, output_path)

# Process each subset (OA, ICA, ICA2)
for subset, task_name in tasks.items():
    subset_path = os.path.join(dataset_root, subset)
    images_dir = os.path.join(subset_path, "imgs")
    masks_dir = os.path.join(subset_path, "masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Skipping {subset}, imgs or masks folder missing")
        continue

    output_root = os.path.join("nnUNet_raw_data", task_name)
    output_image_dir = os.path.join(output_root, "imagesTr")
    output_mask_dir = os.path.join(output_root, "labelsTr")

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".bmp", ".png"))])

    for i, img_file in enumerate(img_files):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, base_name + "_mask.png")

        if not os.path.exists(mask_path):
            print(f"[{subset}] Mask not found for {base_name}, skipping...")
            continue

        out_img_path = os.path.join(output_image_dir, f"{i:04d}_0000.nii.gz")
        out_mask_path = os.path.join(output_mask_dir, f"{i:04d}.nii.gz")

        convert_image_to_nifti(img_path, out_img_path)
        convert_mask_to_nifti(mask_path, out_mask_path)

        print(f"[{subset}] Converted {base_name} → case {i:04d}")
