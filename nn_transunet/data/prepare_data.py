# import os
# import shutil
# import json

# # === Paths to your original data ===
# images_folder = "data/imgs"
# masks_folder = "data/masks"

# # === nnU-Net folder structure ===
# task_id = "001"
# task_name = "MyData"
# task_folder = f"nnUNet_raw/Task{task_id}_{task_name}"
# imagesTr = os.path.join(task_folder, "imagesTr")
# labelsTr = os.path.join(task_folder, "labelsTr")

# # Create directories if they don't exist
# os.makedirs(imagesTr, exist_ok=True)
# os.makedirs(labelsTr, exist_ok=True)

# # === Step 1: Process images ===
# image_files = sorted([f for f in os.listdir(images_folder) if f.endswith((".bmp", ".png"))])
# for img_file in image_files:
#     name = os.path.splitext(img_file)[0]  # e.g., "069_ICA"
#     new_name = f"{name}_0000.bmp"         # single modality convention
#     shutil.copy(os.path.join(images_folder, img_file),
#                 os.path.join(imagesTr, new_name))

# # === Step 2: Process masks ===
# mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith(".png")])
# for mask_file in mask_files:
#     name = os.path.splitext(mask_file)[0]  # e.g., "069_ICA_mask"
#     shutil.copy(os.path.join(masks_folder, mask_file),
#                 os.path.join(labelsTr, f"{name}.png"))

# # === Step 3: Generate dataset.json ===
# dataset_json = {
#     "name": task_name,
#     "description": f"{task_name} dataset converted for nnU-Net",
#     "tensorImageSize": "3D",
#     "reference": "",
#     "licence": "",
#     "release": "0.0",
#     "modality": {
#         "0": "Gray"  # single channel, black and white
#     },
#     "labels": {
#         "0": "background",
#         "1": "foreground"  # modify if you have multiple classes
#     },
#     "numTraining": len(image_files),
#     "numTest": 0,
#     "training": [{"image": f"./imagesTr/{os.path.splitext(f)[0]}_0000.bmp",
#                   "label": f"./labelsTr/{os.path.splitext(f)[0]}_mask.png"} 
#                  for f in image_files],
#     "test": []
# }

# # Save dataset.json
# with open(os.path.join(task_folder, "dataset.json"), "w") as f:
#     json.dump(dataset_json, f, indent=4)

# print(f"Task folder created at {task_folder}")
# print("Images, masks, and dataset.json are ready for nnU-Net preprocessing!")


# import os
# import nibabel as nib
# import numpy as np
# from skimage import io
# from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

# # === CONFIG ===
# RAW_BASE = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_raw"
# TASK_NAME = "Task001_MyData"

# DATASET_DIR = os.path.join(RAW_BASE, TASK_NAME)
# IMAGES_TR = os.path.join(DATASET_DIR, "imagesTr")
# LABELS_TR = os.path.join(DATASET_DIR, "labelsTr")

# # Change these paths to your original PNG/BMP folders
# SOURCE_IMAGES = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/data/imgs"
# SOURCE_LABELS = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/data/masks"

# maybe_mkdir_p(IMAGES_TR)
# maybe_mkdir_p(LABELS_TR)

# def convert_to_nifti(img_path, out_path, is_mask=False):
#     img = io.imread(img_path)
#     # if mask, keep as integer 0/1
#     if is_mask:
#         img = (img > 0).astype(np.uint8)
#     else:
#         # if grayscale, ensure single channel
#         if img.ndim == 2:
#             img = img[np.newaxis, :, :]  # channel-first
#         elif img.ndim == 3 and img.shape[2] == 3:
#             # RGB → grayscale by averaging
#             img = np.mean(img, axis=2, keepdims=True)
#             img = img.transpose(2, 0, 1)  # channel-first
#     # convert to 3D volume with 1 slice if needed
#     if img.ndim == 2:
#         img = img[np.newaxis, :, :]
#     elif img.ndim == 3:
#         pass
#     else:
#         raise ValueError(f"Unexpected image shape: {img.shape}")

#     nii = nib.Nifti1Image(img.astype(np.float32), affine=np.eye(4))
#     nib.save(nii, out_path)
#     print(f"Saved {out_path}")

# # Convert images
# for fname in os.listdir(SOURCE_IMAGES):
#     if fname.lower().endswith((".png", ".bmp")):
#         base = os.path.splitext(fname)[0]
#         out_path = os.path.join(IMAGES_TR, f"{base}_0000.nii.gz")
#         convert_to_nifti(os.path.join(SOURCE_IMAGES, fname), out_path, is_mask=False)

# # Convert masks
# for fname in os.listdir(SOURCE_LABELS):
#     if fname.lower().endswith((".png", ".bmp")):
#         base = os.path.splitext(fname)[0]
#         out_path = os.path.join(LABELS_TR, f"{base}.nii.gz")
#         convert_to_nifti(os.path.join(SOURCE_LABELS, fname), out_path, is_mask=True)

# # === Create dataset.json ===
# from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# # single modality grayscale → channel 0
# generate_dataset_json(
#     DATASET_DIR,
#     modalities={0: "G"},
#     labels={"background": 0, "object": 1},  # rename "object" to whatever your mask represents
#     num_training=len(os.listdir(IMAGES_TR)),
#     file_ending=".nii.gz",
#     dataset_name=TASK_NAME
# )

# print("Dataset ready for nnU-Net preprocessing!")


# import os
# import shutil
# import nibabel as nib
# import numpy as np
# from skimage import io
# from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
# from nnunetv2.paths import nnUNet_raw

# # -------------------------------
# # Configuration
# # -------------------------------

# # Source folders with your current images
# DATA_ROOT = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data"
# SOURCE_IMAGES = os.path.join(DATA_ROOT, "imgs")
# SOURCE_MASKS = os.path.join(DATA_ROOT, "masks")

# # nnU-Net Task folder
# TASK_ID = "001"
# TASK_NAME = "MyData"
# TASK_FOLDER = os.path.join(nnUNet_raw, f"Task{TASK_ID}_{TASK_NAME}")

# # Output folders
# IMAGES_TR = os.path.join(TASK_FOLDER, "imagesTr")
# IMAGES_TS = os.path.join(TASK_FOLDER, "imagesTs")
# LABELS_TR = os.path.join(TASK_FOLDER, "labelsTr")
# LABELS_TS = os.path.join(TASK_FOLDER, "labelsTs")

# os.makedirs(IMAGES_TR, exist_ok=True)
# os.makedirs(IMAGES_TS, exist_ok=True)
# os.makedirs(LABELS_TR, exist_ok=True)
# os.makedirs(LABELS_TS, exist_ok=True)

# # -------------------------------
# # Helper function: convert to NIfTI
# # -------------------------------

# def convert_to_nifti(input_image_path, output_image_path, is_mask=False):
#     img = io.imread(input_image_path)
    
#     if not is_mask:
#         # if RGB, convert to grayscale (sum channels)
#         if len(img.shape) == 3:
#             img = img.sum(axis=2)  # your images are single-channel but just in case
#         img = img.astype(np.float32)
#     else:
#         # mask: ensure binary 0/1
#         img = (img > 0).astype(np.uint8)
    
#     # add dummy z-axis to make 3D (nnU-Net expects 3D)
#     img = np.expand_dims(img, axis=2)
#     nifti_img = nib.Nifti1Image(img, affine=np.eye(4))
#     nib.save(nifti_img, output_image_path)

# # -------------------------------
# # Convert all images
# # -------------------------------

# all_files = sorted(os.listdir(SOURCE_IMAGES))

# for f in all_files:
#     image_path = os.path.join(SOURCE_IMAGES, f)
#     mask_name = f.replace(".bmp", "_mask.png").replace(".png", "_mask.png")
#     mask_path = os.path.join(SOURCE_MASKS, mask_name)
    
#     # Output paths
#     out_image = os.path.join(IMAGES_TR, f.replace(".bmp", "_0000.nii.gz").replace(".png", "_0000.nii.gz"))
#     out_mask = os.path.join(LABELS_TR, mask_name.replace(".png", ".nii.gz"))
    
#     convert_to_nifti(image_path, out_image, is_mask=False)
#     convert_to_nifti(mask_path, out_mask, is_mask=True)

# # -------------------------------
# # Generate dataset.json
# # -------------------------------

# num_training_cases = len(os.listdir(IMAGES_TR))
# modalities = {0: "BW"}  # black & white single-channel
# labels = {"background": 0, "mask": 1}

# generate_dataset_json(
#     TASK_FOLDER,
#     modalities,
#     labels,
#     num_training_cases,
#     file_ending=".nii.gz",
#     dataset_name=f"Task{TASK_ID}_{TASK_NAME}",
#     channel_names=["BW"]
# )

# print(f"Task folder created at {TASK_FOLDER}")
# print("Images, masks, and dataset.json are ready for nnU-Net preprocessing!")

import os
import nibabel as nib
import numpy as np
from skimage import io
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

# === CONFIG ===
RAW_BASE = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_raw"
TASK_NAME = "Task001_MyData"

DATASET_DIR = os.path.join(RAW_BASE, TASK_NAME)
IMAGES_TR = os.path.join(DATASET_DIR, "imagesTr")
LABELS_TR = os.path.join(DATASET_DIR, "labelsTr")

# Change these paths to your original PNG/BMP folders
SOURCE_IMAGES = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/data/imgs"
SOURCE_LABELS = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/data/masks"

maybe_mkdir_p(IMAGES_TR)
maybe_mkdir_p(LABELS_TR)

def convert_to_nifti(img_path, out_path, is_mask=False):
    img = io.imread(img_path)
    # if mask, keep as integer 0/1
    if is_mask:
        img = (img > 0).astype(np.uint8)
    else:
        # if grayscale, ensure single channel
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # channel-first
        elif img.ndim == 3 and img.shape[2] == 3:
            # RGB → grayscale by averaging
            img = np.mean(img, axis=2, keepdims=True)
            img = img.transpose(2, 0, 1)  # channel-first
    # convert to 3D volume with 1 slice if needed
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    elif img.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    nii = nib.Nifti1Image(img.astype(np.float32), affine=np.eye(4))
    nib.save(nii, out_path)
    print(f"Saved {out_path}")

# Convert images
for fname in os.listdir(SOURCE_IMAGES):
    if fname.lower().endswith((".png", ".bmp")):
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(IMAGES_TR, f"{base}_0000.nii.gz")
        convert_to_nifti(os.path.join(SOURCE_IMAGES, fname), out_path, is_mask=False)

# Convert masks
for fname in os.listdir(SOURCE_LABELS):
    if fname.lower().endswith((".png", ".bmp")):
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(LABELS_TR, f"{base}.nii.gz")
        convert_to_nifti(os.path.join(SOURCE_LABELS, fname), out_path, is_mask=True)

# === Create dataset.json ===
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# List your modality names (for single-channel grayscale, just ["G"])
channel_names = {0: "G"}
num_training_cases = len(os.listdir(IMAGES_TR))

generate_dataset_json(
    DATASET_DIR,
    modalities={0: "G"},            # mapping index to modality
    labels={"background": 0, "object": 1},  # your mask labels
    channel_names=channel_names,
    num_training_cases=num_training_cases,
    file_ending=".nii.gz",
    dataset_name=TASK_NAME
)

print("Dataset ready for nnU-Net preprocessing!")
