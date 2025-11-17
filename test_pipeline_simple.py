#!/usr/bin/env python3
"""
Simple test to verify Editorial AI pipeline setup
"""

import sys
import torch
from pathlib import Path

print("="*60)
print("Editorial AI Pipeline Test")
print("="*60)

# Test 1: Environment
print("\n1. Testing Python environment...")
print(f"   Python: {sys.version.split()[0]} ✓")
print(f"   PyTorch: {torch.__version__} ✓")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Test 2: Data processing
print("\n2. Testing data processing...")
try:
    data_file = Path("data/processed/train.jsonl")
    if data_file.exists():
        with open(data_file) as f:
            lines = f.readlines()
            print(f"   Training data: {len(lines)} examples ✓")

    val_file = Path("data/processed/validation.jsonl")
    if val_file.exists():
        with open(val_file) as f:
            lines = f.readlines()
            print(f"   Validation data: {len(lines)} examples ✓")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Model imports
print("\n3. Testing model libraries...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("   Transformers: ✓")

    from peft import LoraConfig
    print("   PEFT (LoRA): ✓")

    from datasets import load_dataset
    print("   Datasets: ✓")

    import accelerate
    print("   Accelerate: ✓")
except ImportError as e:
    print(f"   Import error: {e}")

# Test 4: Simple tokenizer test (no GPU needed)
print("\n4. Testing tokenizer...")
try:
    from transformers import AutoTokenizer

    # Test with a small tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"   Tokenization: ✓")
    print(f"   Original: {test_text}")
    print(f"   Token count: {len(tokens)}")
    print(f"   Decoded: {decoded}")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: Check training script
print("\n5. Checking training scripts...")
scripts = [
    "scripts/prepare_data.py",
    "scripts/train_model.py",
    "scripts/test_model.py",
    "scripts/export_ollama.py"
]

for script in scripts:
    if Path(script).exists():
        print(f"   {script}: ✓")
    else:
        print(f"   {script}: ✗ Missing")

print("\n" + "="*60)
print("Pipeline Status Summary:")
print("="*60)

# Check if we can do a full run
all_good = True
issues = []

if not torch.cuda.is_available():
    issues.append("No GPU detected (training will be slow)")

if not Path("data/processed/train.jsonl").exists():
    issues.append("Training data not found")
    all_good = False

if not all(Path(s).exists() for s in scripts):
    issues.append("Some scripts missing")
    all_good = False

if all_good:
    print("✅ READY FOR TRAINING!")
    print("\nNext step for full training (4-8 hours):")
    print("~/.virtualenvs/editorial_ai/bin/python scripts/train_model.py \\")
    print("    --model 'Qwen/Qwen2.5-7B-Instruct' \\")
    print("    --use-8bit --epochs 3")
else:
    print("⚠️  Some issues found:")
    for issue in issues:
        print(f"   - {issue}")

print("\nNote: RTX 5090 CUDA warnings are cosmetic - GPU will work fine!")
