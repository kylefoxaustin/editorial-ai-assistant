#!/usr/bin/env python3
"""
Simple test script to verify Editorial AI Assistant setup
"""

import sys
import torch

def test_setup():
    """Test the basic setup and GPU access"""

    print("="*60)
    print("Editorial AI Assistant - Setup Test")
    print("="*60)

    # Test Python
    print(f"\n1. Python Version: {sys.version.split()[0]} ✓")

    # Test PyTorch
    print(f"2. PyTorch Version: {torch.__version__} ✓")

    # Test CUDA
    cuda_available = torch.cuda.is_available()
    print(f"3. CUDA Available: {cuda_available} {'✓' if cuda_available else '✗'}")

    if cuda_available:
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"   GPU Name: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")

        # Test CUDA operations
        print("\n4. Testing CUDA Operations...")
        try:
            # Create tensors on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()

            # Perform matrix multiplication
            z = torch.matmul(x, y)

            # Check result
            assert z.shape == (1000, 1000)
            print("   Matrix multiplication: ✓")

            # Test memory allocation
            large_tensor = torch.zeros(10000, 10000).cuda()
            print("   Large tensor allocation: ✓")

            del large_tensor, x, y, z
            torch.cuda.empty_cache()

            print("   GPU Operations: PASSED ✓")

        except Exception as e:
            print(f"   GPU Operations: FAILED ✗")
            print(f"   Error: {e}")
    else:
        print("\n⚠ WARNING: No GPU detected. Training will be very slow!")

    # Test key libraries
    print("\n5. Testing Required Libraries...")

    libraries = {
        "transformers": "Transformers (Hugging Face)",
        "peft": "PEFT (LoRA support)",
        "datasets": "Datasets library",
        "accelerate": "Accelerate (training)",
        "bitsandbytes": "BitsAndBytes (quantization)",
        "gradio": "Gradio (web interface)",
        "pandas": "Pandas (data processing)",
    }

    all_good = True
    for lib, name in libraries.items():
        try:
            __import__(lib)
            print(f"   {name}: ✓")
        except ImportError:
            print(f"   {name}: ✗ MISSING")
            all_good = False

    # Try Unsloth (optional)
    try:
        from unsloth import FastLanguageModel
        print(f"   Unsloth (optimizer): ✓")
    except ImportError:
        print(f"   Unsloth (optimizer): ⚠ Not available (optional)")

    print("\n" + "="*60)

    if cuda_available and all_good:
        print("✅ SETUP COMPLETE - Ready to train!")
        print("\nNext steps:")
        print("1. Prepare your training data in data/raw/")
        print("2. Run: python 1_data_preparation/prepare_dataset.py")
        print("3. Run: python 2_training/train.py")
        return True
    elif all_good:
        print("⚠ SETUP COMPLETE - But no GPU detected")
        print("Training will work but be very slow without a GPU")
        return True
    else:
        print("❌ SETUP INCOMPLETE - Missing required libraries")
        print("Please run: ./setup.sh")
        return False

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
