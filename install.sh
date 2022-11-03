python3 -m venv darwin_venv
source darwin_venv/bin/activate
pip3  install --upgrade pip
mkdir lib;
cd lib;
git clone https://github.com/ultralytics/yolov5;
cd yolov5;
pip3 install -qr requirements.txt
cd ../../
pip3 install rembg[gpu]
 
