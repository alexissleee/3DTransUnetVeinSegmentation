# ===========================
# PowerShell version of nnUNet training script
# ===========================

# Set environment variables
$env:nnUNet_N_proc_DA = 36
$env:nnUNet_codebase = "C:\Users\myname\code\nnUNet"          # replace with your path
$env:nnUNet_raw_data_base = "D:\data\nnUNet_raw_data"     # replace with your dataset
$env:nnUNet_preprocessed = "D:\data\nnUNet_raw_data\nnUNet_preprocessed"
$env:RESULTS_FOLDER = "C:\Users\myname\results"

# Run preprocessing
python D:\code\nnUNet\experiment_planning\nnunet_plan_and_preprocess.py -t 1 --verify_dataset_integrity

# Read CONFIG from script argument
$CONFIG = $args[0]
Write-Host "Using CONFIG: $CONFIG"

# Unit test fold
$fold = 0
Write-Host "Run on fold: $fold"

# Set CUDA devices (comma-separated, same as Linux)
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"

# Use progress bar environment variable
$env:nnunet_use_progress_bar = 1

# Run distributed training
# On Windows, use torchrun instead of python -m torch.distributed.launch
# Make sure your PyTorch version supports torchrun
& python -m torch.distributed.run --nproc_per_node 8 --master_port 4322 `
    "$env:nnUNet_codebase\train.py" --fold=$fold --config=$CONFIG --resume=""
