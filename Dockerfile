# PLAYALTER™ Beast Mode Docker Image
# Version: 2.0.0
# Author: Fatih Ernalbant

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="Fatih Ernalbant"
LABEL version="2.0.0"
LABEL description="PLAYALTER™ Beast Mode - Digital Identity Freedom Platform"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/temp /workspace/outputs /workspace/cache

# Download models at build time (faster cold starts!)
RUN echo "🔥 Downloading AI models..." && \
    wget -q --show-progress -O /workspace/models/inswapper_128.onnx \
    https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx && \
    echo "✅ InSwapper downloaded" && \
    wget -q --show-progress -O /workspace/models/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth && \
    echo "✅ GFPGAN downloaded" && \
    wget -q --show-progress -O /workspace/models/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip -q /workspace/models/buffalo_l.zip -d /workspace/models/ && \
    rm /workspace/models/buffalo_l.zip && \
    echo "✅ Buffalo-L downloaded" && \
    echo "🎉 All models ready!"

# Copy handler
COPY handler.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_PATH=/workspace/models
ENV CACHE_SIZE=100
ENV MAX_VIDEO_LENGTH=60
ENV MAX_WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import handler; print('Health check passed')" || exit 1

# Start command
CMD ["python", "-u", "handler.py"]