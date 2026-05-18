FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# -----------------------------
# Proxy support (important for locked-down network)
# -----------------------------
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# -----------------------------
# Install python + pip
# -----------------------------
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python packages
# -----------------------------
RUN pip3 install --upgrade pip \
    && pip3 install "numpy<2" "opencv-python-headless"

# -----------------------------
# Install PyTorch 1.12.1 + CUDA 11.3
# -----------------------------
RUN pip3 install --no-cache-dir --default-timeout=1000 --progress-bar on \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113