import os
import shutil

# Paths
data_dir = "data"
images_dir = os.path.join(data_dir, "imgs")
masks_dir = os.path.join(data_dir, "masks")

# Output directories
output_base = "dataset"  # will create dataset/OA, dataset/ICA2, dataset/ICA
categories = ["OA", "ICA2", "ICA"]

for cat in categories:
    os.makedirs(os.path.join(output_base, cat, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(output_base, cat, "masks"), exist_ok=True)

# Move images
for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith((".bmp", ".png")):
        continue
    for cat in categories:
        if f"_{cat}" in img_file:
            src = os.path.join(images_dir, img_file)
            dst = os.path.join(output_base, cat, "imgs", img_file)
            shutil.move(src, dst)
            break

# Move masks
for mask_file in os.listdir(masks_dir):
    if not mask_file.lower().endswith(".png"):
        continue
    for cat in categories:
        if f"_{cat}" in mask_file:
            src = os.path.join(masks_dir, mask_file)
            dst = os.path.join(output_base, cat, "masks", mask_file)
            shutil.move(src, dst)
            break

print("Done! Files are now separated into OA, ICA2, ICA subfolders.")
