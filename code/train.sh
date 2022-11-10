export PYTORCH_ENABLE_MPS_FALLBACK=1;python3 ../lib/yolov5/train.py --data ../data/experiments/sample/experiment.yaml  --project ../data/runs/ --device mps --epochs 300

