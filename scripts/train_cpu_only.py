#!/usr/bin/env python3
"""
CPU-only training script - Bypasses all RTX 5090 CUDA issues
This will be slower but will definitely work
"""

import torch
import os

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from PyTorch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="./editorial-ai-model-cpu")
    parser.add_argument("--model", default="gpt2", help="Model to use (gpt2 for testing)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max training steps")
    args = parser.parse_args()

    print(f"Training on CPU (RTX 5090 CUDA issues bypassed)")
    print(f"Using model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model ON CPU
    print("Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 on CPU
        device_map="cpu",  # Force CPU
        low_cpu_mem_usage=True
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print("Device:", next(model.parameters()).device)

    # Load dataset
    print(f"Loading data from {args.data}")
    dataset = load_dataset('json', data_files={
        'train': args.data,
        'validation': args.data.replace('train', 'validation')
    })

    # Simple preprocessing
    def preprocess(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"Edit: {inst}\nOriginal: {inp}\nRevised: {out}"
            texts.append(text)

        result = tokenizer(texts, padding=True, truncation=True, max_length=128)
        result["labels"] = result["input_ids"].copy()
        return result

    print("Preprocessing data...")
    tokenized = dataset.map(preprocess, batched=True)

    # Training arguments for CPU
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10,
        logging_steps=2,
        warmup_steps=5,
        learning_rate=5e-5,
        report_to="none",
        max_steps=args.max_steps,
        no_cuda=True,  # Force CPU
        fp16=False,  # No mixed precision on CPU
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    # Train
    print(f"Starting training for {args.max_steps} steps on CPU...")
    print("This will be slow but will work!")
    trainer.train()

    # Save
    print(f"Saving to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    print("✓ Training complete!")
    print("\nModel saved. You can now:")
    print("1. Test it on CPU")
    print("2. Export to ONNX for better compatibility")
    print("3. Try loading on GPU for inference (might work even if training doesn't)")

if __name__ == "__main__":
    main()
