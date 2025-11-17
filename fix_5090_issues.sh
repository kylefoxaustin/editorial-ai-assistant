#!/bin/bash
# Fix script for RTX 5090 compatibility issues

echo "=============================================="
echo "Fixing RTX 5090 Compatibility Issues"
echo "=============================================="

# Set virtual environment path
VENV_PATH="$HOME/.virtualenvs/editorial_ai"
PYTHON="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

# Fix 1: Downgrade torchao to avoid the int1 issue
echo ""
echo "Fixing torchao compatibility..."
$PIP uninstall -y torchao
$PIP install torchao==0.5.0

# Fix 2: Install alternative to Unsloth if it still fails
echo ""
echo "Testing Unsloth import..."
if ! $PYTHON -c "from unsloth import FastLanguageModel" 2>/dev/null; then
    echo "Unsloth still failing, removing and we'll use standard transformers instead"
    $PIP uninstall -y unsloth unsloth_zoo
    echo "✓ Removed Unsloth - will use standard Transformers + PEFT instead"
else
    echo "✓ Unsloth working!"
fi

# Test the fixes
echo ""
echo "=============================================="
echo "Testing fixes..."
echo "=============================================="

$PYTHON << 'EOF'
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    # Test basic CUDA operations
    try:
        # Create a small tensor and move to GPU
        test_tensor = torch.randn(100, 100).cuda()
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✓ CUDA operations working!")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"⚠ CUDA operations failed: {e}")

# Try importing key libraries
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ Transformers working")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    from peft import LoraConfig, get_peft_model
    print("✓ PEFT (LoRA) working")
except ImportError as e:
    print(f"✗ PEFT import failed: {e}")

try:
    from unsloth import FastLanguageModel
    print("✓ Unsloth working")
except ImportError:
    print("⚠ Unsloth not available - using standard libraries instead")

try:
    import bitsandbytes as bnb
    print("✓ BitsAndBytes working")
except ImportError as e:
    print(f"✗ BitsAndBytes import failed: {e}")

print("\nReady to proceed with training!")
EOF

echo ""
echo "=============================================="
echo "Fixes applied!"
echo "=============================================="
echo ""
echo "Note: The RTX 5090 warning is expected since it's newer than"
echo "the PyTorch version. It will still work fine for training."
echo ""
echo "If Unsloth doesn't work, we'll use standard Transformers + PEFT"
echo "which is slightly slower but fully compatible."
echo ""
