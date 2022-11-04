#!/bin/bash

#run on a single gpu max 6 days
#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=cedric.mesnage@gmail.com

# run the application
source ../../../.profile
source ../darwin_venv/bin/activate
echo "train batch 128 1 gpu"
python3 ../lib/yolov5/train.py --data ../data/experiments/sample/experiment.yaml --project ../data/runs/ --batch 128
