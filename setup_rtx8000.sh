#!/bin/bash
# Complete setup for Editorial AI on RTX 8000

# Create and activate virtual environment
python3 -m venv editorial_env
source editorial_env/bin/activate

# Install all dependencies
pip install torch transformers datasets accelerate peft bitsandbytes pandas

# Create directory structure
mkdir -p data/{raw,processed} scripts models

# Download the sample data and scripts from your previous work
# (You'll need to modify these paths or copy files manually)

echo "✓ Environment ready!"
echo "Now you need to:"
echo "1. Copy your train.jsonl and validation.jsonl to data/processed/"
echo "2. Copy your training scripts to scripts/"
echo "3. Run: python scripts/train_model.py --model 'Qwen/Qwen2.5-7B-Instruct' --use-8bit"
