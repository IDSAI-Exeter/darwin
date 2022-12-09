python3 -m venv darwin_venv
source darwin_venv/bin/activate
pip3  install --upgrade pip
mkdir lib;
cd lib;
git clone https://github.com/ultralytics/yolov5;
cd yolov5;
pip3 install -qr requirements.txt
cd ../../
pip3 install rembg
mkdir data;mkdir data/experiments;mkdir data/runs
mkdir data/animals
pip3 install pandas
cd lib
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
mv azcopy_linux_amd64_10.16.2 azcopy
cd ../