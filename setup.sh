#!/bin/bash
# One-command setup for Editorial AI Assistant

set -e

echo "=============================================="
echo "Setting up Editorial AI Assistant"
echo "=============================================="

# Set up virtual environment paths
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3

# Create virtual environment (if it doesn't exist)
echo ""
if [ ! -d "$WORKON_HOME/editorial_ai" ]; then
    echo "Creating virtual environment..."
    # Use python's venv directly instead of virtualenvwrapper for creation
    python3 -m venv "$WORKON_HOME/editorial_ai"
    echo "✓ Virtual environment created at $WORKON_HOME/editorial_ai"
else
    echo "✓ Virtual environment already exists at $WORKON_HOME/editorial_ai"
fi

# Set paths to use the virtual environment's python and pip directly
VENV_PYTHON="$WORKON_HOME/editorial_ai/bin/python"
VENV_PIP="$WORKON_HOME/editorial_ai/bin/pip"

# Upgrade pip
echo ""
echo "Upgrading pip..."
$VENV_PYTHON -m pip install --upgrade pip

# Check if PyTorch is already installed
echo ""
if $VENV_PYTHON -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch already installed"
    PYTORCH_VERSION=$($VENV_PYTHON -c "import torch; print(torch.__version__)")
    echo "  Version: $PYTORCH_VERSION"

    # Check if it's the CUDA version
    if $VENV_PYTHON -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "  ✓ CUDA support enabled"
    else
        echo "  ⚠ CUDA not available - reinstalling PyTorch with CUDA support..."
        $VENV_PIP uninstall -y torch torchvision torchaudio
        $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    # Install PyTorch with CUDA
    echo "Installing PyTorch with CUDA 12.1..."
    $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install requirements if file exists
if [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing requirements..."
    $VENV_PIP install -r requirements.txt
else
    echo ""
    echo "⚠ Warning: requirements.txt not found, skipping package installation"
fi

# Install Unsloth separately (it's often problematic)
echo ""
echo "Installing Unsloth..."
$VENV_PIP install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || {
    echo "⚠ Unsloth installation failed, trying alternative method..."
    $VENV_PIP install git+https://github.com/unslothai/unsloth.git
}

# Test installation
echo ""
echo "=============================================="
echo "Testing installation..."
echo "=============================================="
$VENV_PYTHON << 'EOF'
import sys
print(f"\n✓ Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Test CUDA functionality
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"✓ CUDA tensor test passed")
        except Exception as e:
            print(f"⚠ CUDA tensor test failed: {e}")
    else:
        print("⚠ No GPU detected - training will be slow!")

except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    from unsloth import FastLanguageModel
    print("✓ Unsloth installed successfully")
except ImportError as e:
    print(f"⚠ Unsloth not installed: {e}")
    print("  This is optional but recommended for faster training")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError:
    print("⚠ Transformers not installed")

try:
    import datasets
    print(f"✓ Datasets library installed")
except ImportError:
    print("⚠ Datasets library not installed")

print("\n" + "="*50)
EOF

# Create activation script for convenience
cat > activate_editorial_ai.sh << 'EOF'
#!/bin/bash
# Convenience script to activate the editorial AI environment
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
workon editorial_ai
EOF

chmod +x activate_editorial_ai.sh

echo ""
echo "=============================================="
echo "Setup complete! 🎉"
echo "=============================================="
echo ""
echo "Virtual environment location: $WORKON_HOME/editorial_ai"
echo ""
echo "To use the environment:"
echo ""
echo "Option 1 - Direct execution (recommended):"
echo "  $VENV_PYTHON your_script.py"
echo ""
echo "Option 2 - Activate in current shell:"
echo "  source activate_editorial_ai.sh"
echo "  python your_script.py"
echo "  deactivate  # when done"
echo ""
echo "Option 3 - Use virtualenvwrapper:"
echo "  workon editorial_ai"
echo "  python your_script.py"
echo "  deactivate  # when done"
echo ""
echo "Next steps:"
echo "  1. Test GPU access: $VENV_PYTHON -c 'import torch; print(torch.cuda.is_available())'"
echo "  2. Run test script: $VENV_PYTHON scripts/test_model.py"
echo ""
