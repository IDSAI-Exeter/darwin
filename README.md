# Darwin

Species identification accuracy optimisation with automated copy-pasting for camera-trap images.

# Configure

edit the config.json file with your jade account, jade ssh key file path, jade home directory and email address.

# Run

sh install.sh

sh download.sh

sh run.sh

Once the training is over :

sh validate.sh

sh plot.sh


# Results of 1,2,4,8 runs are found below

## Training results

![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/montecarlo_1.png?raw=true "training results on 1 image per species")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/montecarlo_2.png?raw=true "training results on 2 image per species")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/montecarlo_4.png?raw=true "training results on 4 image per species")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/montecarlo_8.png?raw=true "training results on 8 image per species")

## Test results

![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/matrix_1_2_4_8_1_2_4_8.png?raw=true "mean test results")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/matrix-std.png?raw=true "mean std error")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/species_1_2_4_8_1_2_4_8.png?raw=true "test results on species")

# Results of 500,1 run

## Training results

![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/montecarlo_500.png?raw=true "training results on 500 images per species")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/matrix_500_1.png?raw=true "mean test results")
![alt text](https://github.com/cedricidsai/darwin/blob/main/plots/species_500_1.png?raw=true "test results on species")
