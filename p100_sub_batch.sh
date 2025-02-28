#!/bin/bash

#SBATCH --job-name=sam2
#SBATCH --account=nn10004k
#SBATCH --output=ouput.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=accel #a100 #accel #normal     # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=3      # Number of CPU cores per task
#SBATCH --time=0-2:15:00               # Maximum runtime (D-HH:MM:SS)
#SBATCH --mem-per-cpu=5G

#SBATCH --gres=gpu:p100:1 

##SBATCH --qos=devel  # for test only 


## Set up job environment:
set -o errexit  # Exit the script on any error

set -o nounset  # Treat any unset variables as an error

# module --quiet purge  # Reset the modules to the system default
module purge --force
# module load  torchvision/0.13.1-foss-2022a-CUDA-11.7.0
source ~/torch_cu124/bin/activate
# nvidia-smi
#print the python path with echo
# which python 
#module list
# python -c "import cv2; print(cv2.__version__)"
# python TRAIN.py
python train.py \
--hiera_path "/cluster/projects/nn10004k/packages_install/sam_checkpoints/sam2_hiera_large.pt" \
--train_image_path "/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch896_exclude_ukan/TYPE2/images/" \
--train_mask_path "/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch896_exclude_ukan/TYPE2/masks/3/" \
--save_path "/cluster/projects/nn10004k/packages_install/sam2_results" \
--epoch 1 \
--lr 0.001 \
--batch_size 4