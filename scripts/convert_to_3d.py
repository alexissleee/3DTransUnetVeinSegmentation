import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import os
import h5py

def construct_volume_from_slices(slices_folder: str, output_path: str,  volume_name: str):
    slices_folder = slices_folder.strip('/')

    image_paths = [os.path.join(slices_folder, filename) 
                   for filename in sorted(os.listdir(slices_folder))]

    images = [np.array(Image.open(path), dtype=np.float32) for path in image_paths]

    volume_data = np.stack(images, axis=0)

    volume_data = np.clip(volume_data, -125, 275)

    v_min, v_max = volume_data.min(), volume_data.max()
    if v_max != 0: 
        volume_data = (volume_data - v_min) / (v_max - v_min)
    else:
        volume_data = np.zeros_like(volume_data)
    
    os.makedirs(output_path, exist_ok=True)
    with h5py.File(os.path.join(output_path, volume_name), 'w') as hf:
        hf.create_dataset(volume_name, data=volume_data, compression="gzip")
        print(f"Successfully created HDF5 volume at '{output_path}/{volume_name} with dataset {volume_name}")
        


if __name__ == '__main__':
    # img_folder = './raw_data/OA/imgs'
    # label_folder = './raw_data/OA/masks'

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the folder containing BMP slices')
    args = parser.parse_args()

    name = args.name
    imgs_folder = Path(args.name + "/imgs")
    masks_folder = Path(args.name + "/masks")
    output_path = Path(args.output)
    construct_volume_from_slices(slices_folder=imgs_folder, output_path="./processed_data/" + name + "/volumes/", volume_name=name+"_images.h5")
    construct_volume_from_slices(slices_folder=masks_folder, output_path="./processed_data/" + name + "/volumes/", volume_name=name+"_labels.h5")
