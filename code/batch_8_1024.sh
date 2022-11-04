#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=cedric.mesnage@gmail.com

# run the application
source ../../../.profile
source ../darwin_venv/bin/activate
echo "python3 -m torch.distributed.run --nproc_per_node 8 ../lib/yolov5/train.py --batch 1024 --data ../data/experiments/sample/experiment.yaml --project ../data/runs/ --device 0,1,2,3,4,5,6,7"
python3 -m torch.distributed.run --nproc_per_node 8 ../lib/yolov5/train.py --batch 1024 --data ../data/experiments/sample/experiment.yaml --project ../data/runs/ --device 0,1,2,3,4,5,6,7
