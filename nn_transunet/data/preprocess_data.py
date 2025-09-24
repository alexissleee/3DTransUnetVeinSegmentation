import os
import argparse
from nn_transunet.data.preprocessing import PreprocessorFor2D

parser = argparse.ArgumentParser(description="Run nnU-Net 2D preprocessing for a task")
parser.add_argument("task_name", type=str, help="Task name (e.g., 001_OA)")
args = parser.parse_args()

task_name = args.task_name

dataset_dir = os.path.join("nnUNet_raw_data_base", task_name)
output_dir = os.path.join("nnUNet_preprocessed", task_name)

os.makedirs(output_dir, exist_ok=True)

# Initialize preprocessor
preprocessor = PreprocessorFor2D(
    plans_file=None,  # use default plans
    dataset_directory=dataset_dir,
    preprocessing_output_dir=output_dir,
    num_threads_preprocessing=1,
    num_processes_preprocessing=1,
)

# Run preprocessing
preprocessor.run()
print(f"Preprocessing complete for task {task_name}")
