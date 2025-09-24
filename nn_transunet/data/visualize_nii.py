import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

def visualize_nii_volume(nii_path: str):
    img = nib.load(nii_path)
    volume = img.get_fdata()
    depth = volume.shape[2]  # assume slices along Z-axis

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    slice_idx = 0
    img_plot = ax.imshow(volume[:, :, slice_idx].T, cmap='gray', origin='lower')
    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, depth - 1, valinit=slice_idx, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_plot.set_data(volume[:, :, idx].T)
        ax.set_title(f"Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a NIfTI volume")
    parser.add_argument("nii_path", type=str, help="Path to the .nii.gz file")
    args = parser.parse_args()

    visualize_nii_volume(args.nii_path)
