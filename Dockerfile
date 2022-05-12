ARG BASE_IMAGE=ubuntu:18.04
ARG CUDA_VERSION=11.4
ARG CUDA=11.4
ARG CUDNN_VERSION=8
ARG PYTHON_VERSION=3.8
ARG CONDA_ENV_NAME=plightning
#ARG UID=
#ARG USER_NAME=
#LABEL maintainer "onlyred"

# Needed for string substitution
FROM ${BASE_IMAGE} as dev-base
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    vim \
    libgl1-mesa-glx \
    wget && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:${PATH}

# Install miniconda
FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda clean -ya

# Create the conda environment
RUN /opt/conda/bin/conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION
ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Install the packages
RUN source activate ${CONDA_ENV_NAME} && \
    /opt/conda/bin/conda install pytorch torchvision cudatoolkit=${CUDA} -c pytorch
RUN source activate ${CONDA_ENV_NAME} && \
    /opt/conda/bin/conda install pytorch-lightning \
                                 scikit-learn \
                                 pandas \
                                 opencv \
                                 tqdm \
                                 kornia \
                                 -c conda-forge
