import os
import numpy as np
from PIL import Image
import argparse

# Mapping subfolders to task numbers (or names)
tasks = {
    "OA": "001_OA",
    "ICA": "002_ICA",
    "ICA2": "003_ICA2"
}

def bmp_to_npz_folder(input_folder, output_file):
    slices = []
    bmp_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".bmp")])
    for filename in bmp_files:
        path = os.path.join(input_folder, filename)
        img = Image.open(path).convert("L")  # grayscale
        arr = np.array(img)
        slices.append(arr)
    if slices:
        stacked = np.stack(slices)  # (num_slices, H, W)
        np.savez_compressed(output_file, images=stacked)
        print(f"Saved {output_file} with shape {stacked.shape}")
    else:
        print(f"No BMPs found in {input_folder}, skipping.")

def main(dataset_root, output_root):
    for subset, task_name in tasks.items():
        subset_path = os.path.join(dataset_root, subset)
        images_dir = os.path.join(subset_path, "imgs")

        if not os.path.exists(images_dir):
            print(f"Skipping {subset}, imgs folder missing")
            continue

        output_task_dir = os.path.join(output_root, task_name)
        os.makedirs(output_task_dir, exist_ok=True)

        bmp_folders = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))])
        if not bmp_folders:
            bmp_folders = [images_dir]  # in case images are directly in imgs/

        for i, folder in enumerate(bmp_folders):
            output_file = os.path.join(output_task_dir, f"{i:04d}.npz")
            bmp_to_npz_folder(folder, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Root folder containing subsets (OA, ICA, ICA2)")
    parser.add_argument("--output_root", type=str, default="nnUNet_raw_data_base", help="Root folder to save NPZ files")
    args = parser.parse_args()

    main(args.dataset_root, args.output_root)
