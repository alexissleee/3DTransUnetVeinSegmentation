
export nnUNet_N_proc_DA=36
export nnUNet_codebase="/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet" # replace to your codebase
export nnUNet_raw_data_base="/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_raw_data_base/" # replace to your database
export nnUNet_preprocessed="/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/nnUNet_preprocessed"
export RESULTS_FOLDER="/Users/alee/UCLA/Research/IRISS/3D-TransUNet/nn_transunet/data/results/"


CONFIG=$1

echo $CONFIG

### unit test
fold=0
echo "run on fold: ${fold}"
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=8 \
        ./train.py --fold=${fold} --config=$CONFIG --resume=''