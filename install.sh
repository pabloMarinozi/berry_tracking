conda create --name berry_tracking python=3.7
conda activate berry_tracking
conda install pytorch=0.4.1 cuda92 torchvision -c pytorch
PYTORCH=/home/pablo/miniconda3/envs/berry_tracking/lib/python3.7/site-packages #cambiar por la ruta que contenga la carpeta torch en su máquina
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
conda install -c anaconda cmake
conda install -c anaconda cython
pip install -r detector/requirements.txt
pip install -r requirements.txt
cd detector/models/networks/DCNv2/
chmod a+x make.sh
./make.sh
cd ../../../external/
make
pip install gdown
cd ../../
gdown https://drive.google.com/uc?id=1XsvtpexSGl8kwA1xgeCUBpyOIuSvXy8V #cambiar con el modelo correspondiente


