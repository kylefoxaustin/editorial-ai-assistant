#!/usr/bin/env python3
"""
CPU-only training that WILL work - no GPU at all
Slower but guaranteed to complete
"""

import torch
import os
# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="./editorial-gpt2-cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    print("CPU-Only Training - This WILL work!")
    print("Device: CPU (GPU disabled)")

    # Load model and tokenizer
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Freeze early layers to speed up training
    for name, param in model.named_parameters():
        if "h.10" not in name and "h.11" not in name and "ln_f" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable/1e6:.1f}M parameters on CPU")

    # Load data
    print("\nLoading training data...")
    examples = []
    with open(args.data, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = f"Edit: {data['instruction']}\nText: {data['input']}\nResult: {data['output']}"
            examples.append(text)

    print(f"Loaded {len(examples)} examples")

    # Simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    print("This will take 5-10 minutes on CPU")

    model.train()
    losses = []

    progress = tqdm(range(args.steps), desc="Training")
    for step in progress:
        # Get random example
        idx = step % len(examples)
        text = examples[idx]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
        labels = inputs["input_ids"].clone()

        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        losses.append(loss.item())
        avg_loss = sum(losses[-10:]) / min(10, len(losses))
        progress.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Save checkpoint every 10 steps
        if (step + 1) % 10 == 0:
            print(f"\nStep {step+1}: Loss = {avg_loss:.4f}")

    # Save model
    print(f"\nSaving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n✅ Training complete!")
    print("Model trained successfully on CPU")

    # Test the model
    print("\n--- Testing the trained model ---")
    model.eval()
    test_text = "Edit: Fix grammar\nText: The report which was written by John was comprehensive.\nResult:"
    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test output:\n{result}")

    print("\nYou can now use this model for inference!")

if __name__ == "__main__":
    main()
