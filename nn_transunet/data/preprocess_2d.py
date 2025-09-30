import os
import subprocess

# Set nnU-Net paths
os.environ["nnUNet_raw_data_base"] = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_preprocessed"
os.environ["RESULTS_FOLDER"] = "/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_trained_models"

# Run preprocessing via CLI
subprocess.run(["nnUNet_plan_and_preprocess", "-t", "1"], check=True)
