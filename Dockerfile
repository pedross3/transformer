# ------------------------------------------------------------
# TensorFlow 1.15 + CUDA 10.0 + cuDNN 7 on Ubuntu 18.04
# ------------------------------------------------------------
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# --- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6 python3.6-dev python3-pip python3.6-venv \
    build-essential git wget curl ca-certificates \
    libfreetype6-dev libpng-dev libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Pip setup --------------------------------------------------------------
RUN python3.6 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- TensorFlow 1.15 GPU + extras ------------------------------------------
RUN pip install --no-cache-dir \
    tensorflow-gpu==1.15 \
    tensorflow-graphics==2021.12.3 \
    numpy==1.18.5 \
    matplotlib==3.3.4 \
    pandas==1.1.5 \
    scipy==1.4.1 \
    scikit-learn==0.22.2 \
    seaborn==0.11.0

# --- Working directory ------------------------------------------------------
WORKDIR /app
COPY . /app

# Optional: create output dirs
RUN mkdir -p checkpoints outputs

ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["/bin/bash"]
