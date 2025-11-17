#!/usr/bin/env python3
"""
Nanochat-style training for Editorial AI
No LoRA, no fancy casting - just direct training that works on RTX 5090
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import argparse
from tqdm import tqdm

class EditorialDataset(Dataset):
    """Simple dataset for editorial training"""

    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Create training prompt
                text = f"Instruction: {data['instruction']}\nInput: {data['input']}\nOutput: {data['output']}"
                self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()  # For causal LM
        }

def train_model(model, train_loader, val_loader, device, epochs=1, lr=1e-5):
    """Simple training loop like nanochat"""

    # Move model to device
    model = model.to(device)
    model.train()

    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            try:
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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Update stats
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

                # Save checkpoint every 10 steps
                if (batch_idx + 1) % 10 == 0:
                    print(f"\nStep {batch_idx+1}: Loss = {avg_loss:.4f}")
                    # Save checkpoint
                    torch.save({
                        'step': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, f'checkpoint_step_{batch_idx+1}.pt')

            except RuntimeError as e:
                print(f"\nError at step {batch_idx}: {e}")
                print("Attempting to continue...")
                continue

        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model to fine-tune")
    parser.add_argument("--data", default="data/processed/train.jsonl")
    parser.add_argument("--val-data", default="data/processed/validation.jsonl")
    parser.add_argument("--output", default="./editorial-ai-5090")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-mode", action="store_true", help="Quick test with 5 examples")

    args = parser.parse_args()

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Load tokenizer and model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model directly - no LoRA, no quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 for stability
        low_cpu_mem_usage=True
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Freeze most layers to save memory (optional - comment out to train everything)
    if args.model == "gpt2":
        # For GPT-2, only train last 2 transformer blocks
        for name, param in model.named_parameters():
            if "h.10" not in name and "h.11" not in name and "ln_f" not in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {trainable_params/1e6:.1f}M parameters")

    # Create datasets
    print("\nLoading data...")
    train_dataset = EditorialDataset(args.data, tokenizer, args.max_length)
    val_dataset = EditorialDataset(args.val_data, tokenizer, args.max_length)

    if args.test_mode:
        # Use only first 5 examples for testing
        train_dataset.examples = train_dataset.examples[:5]
        val_dataset.examples = val_dataset.examples[:2]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # No multiprocessing to avoid issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Train
    print("\nStarting training...")
    print("If this crashes with CUDA errors, we'll add more workarounds")

    try:
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr
        )

        # Save final model
        print(f"\nSaving model to {args.output}")
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

        print("✓ Training complete!")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("Saving model anyway...")
        try:
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print("Model saved despite errors")
        except:
            print("Could not save model")

if __name__ == "__main__":
    main()
