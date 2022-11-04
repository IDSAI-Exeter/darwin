#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=cedric.mesnage@gmail.com

# run the application
source ../../../.profile
source ../darwin_venv/bin/activate
echo "python3 -m torch.distributed.run --nproc_per_node 2 ../lib/yolov5/train.py --batch 64 --data ../data/experiments/sample/experiment.yaml --project ../data/runs/ --device 0,1"
python3 -m torch.distributed.run --nproc_per_node 2 ../lib/yolov5/train.py --batch 64 --data ../data/experiments/sample/experiment.yaml --project ../data/runs/ --device 0,1
