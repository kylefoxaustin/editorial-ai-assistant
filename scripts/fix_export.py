#!/usr/bin/env python3
"""
Export Editorial AI Model to Ollama - Fixed Version
"""

import os
import subprocess
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_weights(model_path, base_model, output_path):
    """Merge LoRA weights with base model"""
    
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Export model to Ollama")
    parser.add_argument("--model", default="./editorial-ai-model-full", help="Fine-tuned model path")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--output", default="./merged-editorial-model", help="Output path")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    # Check if it's a LoRA model
    if (model_path / "adapter_config.json").exists():
        print("Detected LoRA adapter, merging with base model...")
        print("This will take a few minutes...")
        merged_path = Path(args.output)
        merge_lora_weights(model_path, args.base_model, merged_path)
        print(f"\n✅ Success! Merged model saved to {merged_path}")
        print("\nYour model is ready to use!")
        print(f"Test it with: python scripts/test_trained_model.py")
        print("\nFor Ollama conversion, you'll need llama.cpp to convert to GGUF format")
    else:
        print("Not a LoRA model, skipping merge")

if __name__ == "__main__":
    main()
