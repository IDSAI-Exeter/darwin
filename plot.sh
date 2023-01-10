source darwin_venv/bin/activate;
cd src;
python3 plot.py -e "projects/darwin/data/experiments/montecarlo/" -k 5
python3 matrix.py -e "projects/darwin/data/experiments/montecarlo/" -a 1,2,4,8 -r 1,2,4,8 -k 5
