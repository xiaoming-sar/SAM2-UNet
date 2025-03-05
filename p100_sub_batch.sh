#!/bin/bash

#SBATCH --job-name=sam2
#SBATCH --account=nn10004k
#SBATCH --output=ouput.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=accel #accel #normal  a100   # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=3      # Number of CPU cores per task
#SBATCH --time=0-2:15:00               # Maximum runtime (D-HH:MM:SS)
#SBATCH --mem-per-cpu=5G

##SBATCH --gres=gpu:a100:1 
#SBATCH --gres=gpu:p100:1 
##SBATCH --qos=devel  # for test only 


## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
source ~/torch_cu124/bin/activate
# module load  torchvision/0.13.1-foss-2022a-CUDA-11.7.2
nvidia-smi
#print the python path with echo
# which python 
#module list
# python -c "import cv2; print(cv2.__version__)"
# python TRAIN.py
# the maximum batch size is 1 for p100
# Define your paths as variables


python train_mod.py  \
--hiera_path "/cluster/projects/nn10004k/packages_install/sam_checkpoints/sam2_hiera_large.pt" \
--data_dir "/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch896_exclude_ukan/" \
--save_path "/cluster/projects/nn10004k/packages_install/sam2_results" \
--im_size 896 --num_classes 4 --batch_size 1 --epoch 1 --lr 0.001 