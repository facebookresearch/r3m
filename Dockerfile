ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=11.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.0.4.30-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.1.3-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Needed for string substitution
SHELL ["/bin/bash", "-l", "-c"]

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# bug (4 Aug 2020)
RUN ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11


# pyenv (from [2]) ----------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN curl https://pyenv.run | bash && \
    echo '' >> /root/.bash_profile && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.bash_profile && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bash_profile && \
    echo 'eval "$(pyenv init --path)"' >> /root/.bash_profile && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.bash_profile
RUN source /root/.bash_profile && \
    pyenv install anaconda3-2022.05 && \
    pyenv global anaconda3-2022.05 && \
    pip install -U pip

# X window ----------------
RUN apt-get update && apt-get install -y \
    xvfb x11vnc python-opengl icewm
RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 & x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 & icewm-session &"' >> /root/.bashrc && \
    echo 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-460/libGL.so' >> /root/.bashrc

# DL libraries and jupyter ----------------
RUN source /root/.bash_profile && \
    pip install setuptools jupyterlab && \
    pip install tensorflow && \
    pip install pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    echo 'alias jl="jupyter lab --ip 0.0.0.0 --port 8888 --NotebookApp.token='' --allow-root &"' >> /root/.bashrc && \
    echo 'alias tb="tensorboard --host 0.0.0.0 --port 6006 --logdir runs &"' >> /root/.bashrc

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev

# utils ----------------
RUN apt-get update && apt-get install -y \
    vim

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root/workspace

COPY ./r3m/r3m_base.yaml /root/workspace/
RUN source /root/.bash_profile && \
    pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda env create -f r3m_base.yaml && \
    conda init && \
    echo "conda activate r3m_base" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV r3m_base && \
    PATH /opt/conda/envs/r3m_base/bin:$PATH

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

WORKDIR /
RUN git clone https://github.com/openai/mujoco-py.git -b v2.1.2.14 --depth 1 && \
    cd mujoco-py && \
    source /root/.bash_profile && \
    pip install -r requirements.txt && \
    sed -i -e 's/= LinuxCPU/= LinuxGPU/g' mujoco_py/builder.py && \
    pip install -e . && \
    cd && \
    pip install gym==0.21.0 wheel moviepy opencv-python opencv-contrib-python wandb mujoco_py

# RUN conda install pandas libgcc && \
#     echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.pyenv/versions/anaconda3-2022.05/envs/r3m_base/lib' >> /root/.bashrc

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> /root/.bashrc && \
    echo 'test -r ~/.bashrc && . ~/.bashrc' >> /root/.bash_profile

RUN cd / && \
    git clone https://github.com/suraj-nair-1/metaworld.git && \
    cd metaworld && \
    source /root/.bash_profile && \
    sed -i -e 's/mujoco-py<2.1,>=2.0/mujoco-py/' setup.py && \
    pip install -e .

WORKDIR /root/workspace/dataset
RUN wget https://drive.google.com/drive/folders/108VW5t5JV8uNtkWvfZxEvY2P2QkC_tsf?usp=sharing

WORKDIR /root/workspace/
RUN source ~/.bash_profile && source ~/.bashrc
CMD ["/bin/bash", "-c", "source ~/.bash_profile && bash"]
