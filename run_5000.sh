source darwin_venv/bin/activate
cd src/
python3 montecarlo.py -e ../data/experiments/montecarlo_5000 -r 100,300,500 -a 1,2,4 -n 100 -s baboon,eland,elephant,lionfemale,topi,wildebeest
