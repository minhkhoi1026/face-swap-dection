FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Package version control

ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=10.1
ARG TENSORFLOW_VERSION=2.2.0
ARG CUDA_CHANNEL=nvidia

# Setup workdir and non-root user

ARG USERNAME=hcmus
WORKDIR /home/$USERNAME/workspace/

RUN apt-get update &&\
    apt-get install -y --no-install-recommends curl git sudo &&\
    useradd --create-home --shell /bin/bash $USERNAME &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME &&\
    rm -rf /var/lib/apt/lists/*


RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get -qq update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    gcc \
    tmux \
    libjpeg-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*


# # Install conda
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    chown -R $USERNAME:$USERNAME /opt/conda/ && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    tensorflow=${TENSORFLOW_VERSION} "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya

# Set up environment variables
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# # # Install repo dependencies
SHELL ["/bin/bash", "--login", "-c"] 
COPY requirements.txt $WORKDIR
RUN conda init bash && source ~/.bashrc && conda activate && \
    python -m pip install -r requirements.txt

USER $USERNAME
RUN conda init bash 
