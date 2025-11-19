#!/bin/bash

echo "Checking for GPU support..."

# Check if nvidia-docker is available
if command -v nvidia-docker &> /dev/null || docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null 2>&1; then
    echo "✓ GPU support available"
    docker run --rm -it --gpus all \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        -p 11434:11434 \
        editorial-ai-assistant:latest
else
    echo "⚠ No GPU support - running CPU only"
    docker run --rm -it \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        -p 11434:11434 \
        editorial-ai-assistant:latest
fi
