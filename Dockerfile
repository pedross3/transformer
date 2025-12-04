# Use NVIDIA CUDA base image with Python 3.6 support
# TensorFlow 2.6.2 requires CUDA 11.2 and cuDNN 8.1
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:${PATH}" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python3.6 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Install numpy first as some packages depend on it
RUN pip install --no-cache-dir numpy==1.19.5

# Install TensorFlow with GPU support
RUN pip install --no-cache-dir tensorflow-gpu==2.6.2

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for checkpoints and outputs if they don't exist
RUN mkdir -p checkpoints outputs

# Set Python path to include the current directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose port for TensorBoard (optional)
EXPOSE 6006

# Default command (can be overridden)
CMD ["/bin/bash"]