# Use NVIDIA CUDA base image for GPU support (works for CPU too)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements-docker.txt .
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir -r requirements-docker.txt || \
    pip3 install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    pandas \
    numpy \
    sentencepiece \
    tqdm

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy application files
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p models data/processed

# Copy startup script
COPY docker_startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

ENTRYPOINT ["/app/startup.sh"]
