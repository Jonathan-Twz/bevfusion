FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda (using older version to avoid ToS issues with Python 3.8)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
# Use conda-forge to avoid Anaconda ToS requirements
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda config --remove channels defaults
RUN conda install -y python=3.8
# Install PyTorch via pip for better compatibility (avoid MKL symbol issues)
# RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack==0.3.1
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0

# category of map
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numpy==1.23.5
RUN pip install numba==0.48.0

# Fix compatibility issues for nuCarla training
RUN pip install yapf==0.32.0
RUN pip install setuptools==59.5.0
