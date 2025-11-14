#!/bin/bash
# One-command setup for Editorial AI Assistant

set -e

echo "=============================================="
echo "Setting up Editorial AI Assistant"
echo "=============================================="

# Check virtualenvwrapper
if ! command -v mkvirtualenv &> /dev/null; then
    echo "Installing virtualenvwrapper..."
    pip install virtualenvwrapper
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
mkvirtualenv -p python3.10 editorial_ai

# Install PyTorch
echo ""
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To get started:"
echo "  workon editorial_ai"
echo "  python scripts/test_model.py"
echo ""
