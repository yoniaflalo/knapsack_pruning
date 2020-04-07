FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
#FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN mkdir /app
WORKDIR /app



RUN apt-get update && apt-get -y upgrade
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
ADD requirements.txt ./
RUN pip install -r requirements.txt

RUN git clone https://github.com/mapillary/inplace_abn.git
RUN cd inplace_abn && python setup.py install
RUN python3 -c "from inplace_abn import InPlaceABN;"

