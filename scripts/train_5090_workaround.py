#!/usr/bin/env python3
"""
RTX 5090 Compatible Training - Avoids problematic CUDA operations
Works around torch.arange and similar issues
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

class EditorialDataset:
    """Dataset that avoids problematic CUDA operations"""

    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = f"Instruction: {data['instruction']}\nInput: {data['input']}\nOutput: {data['output']}"
                # Pre-tokenize on CPU to avoid CUDA issues
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                self.examples.append({
                    'input_ids': encoded['input_ids'].squeeze().numpy(),  # Store as numpy
                    'attention_mask': encoded['attention_mask'].squeeze().numpy()
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def train_step(model, batch, optimizer, device):
    """Single training step with error handling"""

    # Convert numpy to tensor on CPU first, then move to GPU
    input_ids = torch.from_numpy(batch['input_ids']).unsqueeze(0)
    attention_mask = torch.from_numpy(batch['attention_mask']).unsqueeze(0)

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = input_ids.clone()

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2", help="Model to fine-tune")
    parser.add_argument("--data", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="./editorial-ai-5090-workaround")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use-gpt2", action="store_true", help="Use GPT-2 for testing")

    args = parser.parse_args()

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Note: Using workarounds for RTX 5090 compatibility")

    # Select model
    if args.use_gpt2:
        model_name = "gpt2"
        print("Using GPT-2 for testing")
    else:
        model_name = args.model

    # Load tokenizer and model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with specific settings to avoid issues
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        # Don't use device_map auto - we'll move manually
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Selectively freeze layers
    if "gpt2" in model_name:
        for name, param in model.named_parameters():
            if "h.10" not in name and "h.11" not in name and "ln_f" not in name:
                param.requires_grad = False
    elif "phi" in model_name:
        # Freeze early layers of Phi-2
        for name, param in model.named_parameters():
            if "layers.0" in name or "layers.1" in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable_params/1e6:.1f}M parameters")

    # Move model to device
    print("Moving model to GPU...")
    model = model.to(device)
    model.train()

    # Create dataset
    print("\nLoading and pre-tokenizing data...")
    dataset = EditorialDataset(args.data, tokenizer)
    print(f"Loaded {len(dataset)} examples")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    losses = []

    progress = tqdm(range(args.steps), desc="Training")
    for step in progress:
        try:
            # Get example
            idx = step % len(dataset)
            batch = dataset[idx]

            # Train step
            loss = train_step(model, batch, optimizer, device)
            losses.append(loss)

            # Update progress
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            progress.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Save checkpoint
            if (step + 1) % 20 == 0:
                print(f"\nStep {step+1}: Loss = {avg_loss:.4f}")

        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"\nCUDA error at step {step}: {e}")
                print("Continuing...")
                continue
            else:
                raise e

    # Save model
    print(f"\nSaving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("✓ Training complete!")

    # Quick test
    print("\nTesting model...")
    model.eval()
    test_text = "Instruction: Fix grammar\nInput: The report was comprehensive.\nOutput:"

    # Tokenize on CPU
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Use CPU for generation to avoid CUDA issues
        model_cpu = model.cpu()
        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
        outputs = model_cpu.generate(**inputs_cpu, max_new_tokens=30, do_sample=True, temperature=0.7)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {result}")

if __name__ == "__main__":
    main()
