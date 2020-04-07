FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
#FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN mkdir /app
WORKDIR /app

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list
COPY Docker/pip.conf /etc/pip.conf

# libwebp support + libjpeg-turbo library + custom fork of pillow-simd that has NAME=Pillow (so that transitive dependencies that require Pillow don't override it)
RUN apt-get update && apt-get -y upgrade
#RUN apt-get update && apt-get install -y libwebp-dev && \
#    conda update -n base -c defaults conda && \
#    conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo && \
#    pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo && \
#    conda install -yc conda-forge libjpeg-turbo && \
#    conda install -yc zegami libtiff-libjpeg-turbo && \
#    CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile git+https://code.aliyun.com/icvl/pillow-simd.git && \
#    conda install -y jpeg libtiff certifi

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#RUN cd apex && pip install -v --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Cache the dependencies for faster builds
#RUN pip install pybind11 && pip install bayesian-optimization dill flask graphviz git+https://code.aliyun.com/hyperml/hyperml-python-sdk.git@2.5.0 git+https://github.com/Lyken17/pytorch-OpCounter.git imageio matplotlib numpy>=1.14.3 opencv-python oss2 pybind11 pytest>=4.2.0 pytest-cov==2.5.1 pytest-forked pytest-html pytest-profiling pytest-xdist tensorboardX yacs opencv-python flask plotly nmslib # Deployment requirements https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/apps/eas/allspark-0.6-py2.py3-none-any.whl https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/apps/eas/eascmd-1.5-py2.py3-none-any.whl torch==1.1.0
RUN pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
ADD requirements.txt ./
RUN pip install -r requirements.txt

RUN git clone https://github.com/mapillary/inplace_abn.git
RUN cd inplace_abn && python setup.py install
RUN python3 -c "from inplace_abn import InPlaceABN;"

