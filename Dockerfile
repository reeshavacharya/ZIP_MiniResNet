# GPU-enabled base image with CUDA 12.8 runtime (adjust tag if needed)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps: Python, pip, build tools, Git, Go (for zk-Location), curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential git curl ca-certificates \
        golang-go && \
    rm -rf /var/lib/apt/lists/*

# Make python3 the default `python`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies matching your current virtual environment
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Default working directory inside the container
WORKDIR /app

# By default, just show Python & CUDA versions; override CMD when running.
CMD ["bash", "-lc", "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"]
