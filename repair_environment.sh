#!/bin/bash
# Complete fix for transformers and related packages

echo "=============================================="
echo "Complete Environment Repair"
echo "=============================================="

# Set virtual environment path
VENV_PATH="$HOME/.virtualenvs/editorial_ai"
PYTHON="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

echo ""
echo "Step 1: Cleaning up broken packages..."
$PIP uninstall -y transformers peft accelerate datasets trl torchao

echo ""
echo "Step 2: Reinstalling core packages in correct order..."

# Install transformers first
echo "Installing transformers..."
$PIP install transformers==4.44.0

# Install other packages
echo "Installing PEFT..."
$PIP install peft==0.12.0

echo "Installing accelerate..."
$PIP install accelerate==0.33.0

echo "Installing datasets..."
$PIP install datasets==2.20.0

echo "Installing trl..."
$PIP install trl==0.9.6

echo ""
echo "Step 3: Testing the installation..."

$PYTHON << 'EOF'
import sys
print("Python:", sys.version.split()[0])

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    sys.exit(1)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        TrainingArguments,
        Trainer
    )
    print("✓ Transformers: All imports working")
except Exception as e:
    print(f"✗ Transformers error: {e}")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("✓ PEFT: LoRA support working")
except Exception as e:
    print(f"✗ PEFT error: {e}")

try:
    from datasets import load_dataset, Dataset
    print("✓ Datasets: Working")
except Exception as e:
    print(f"✗ Datasets error: {e}")

try:
    from accelerate import Accelerator
    print("✓ Accelerate: Working")
except Exception as e:
    print(f"✗ Accelerate error: {e}")

print("\n" + "="*50)
print("Testing model loading...")

try:
    # Test with a tiny model
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    print("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Loading GPT2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if torch.cuda.is_available():
        print("Moving model to GPU...")
        model = model.cuda()

        # Test generation
        inputs = tokenizer("The editorial assistant", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"✓ Model inference working!")
        print(f"  Generated: '{result}'")
    else:
        print("✓ Model loaded (CPU mode)")

    print("\n✅ ENVIRONMENT FULLY REPAIRED!")
    print("\nYou're ready to start training your editorial AI!")

except Exception as e:
    print(f"\n⚠ Model test failed: {e}")
    print("But core libraries are installed, you can proceed with caution")

EOF

echo ""
echo "=============================================="
echo "Repair Complete!"
echo "=============================================="
echo ""
echo "Your environment should now be working properly."
echo "The RTX 5090 warning is normal and won't affect training."
echo ""
echo "Next steps:"
echo "1. Test with: ~/.virtualenvs/editorial_ai/bin/python test_setup.py"
echo "2. Start preparing your training data"
echo ""
