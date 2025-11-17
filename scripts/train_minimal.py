#!/usr/bin/env python3
"""
Minimal training script for RTX 5090 - No LoRA, just basic fine-tuning
Warning: This will use more memory but avoids compatibility issues
"""

import torch
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
    parser.add_argument("--output", default="./editorial-ai-model-simple")
    parser.add_argument("--test-mode", action="store_true", help="Quick test with tiny model")
    args = parser.parse_args()

    # For testing, use a tiny model
    if args.test_mode:
        model_name = "gpt2"  # 124M params, good for testing
        print("TEST MODE: Using GPT-2 for quick testing")
    else:
        model_name = "microsoft/phi-2"  # 2.7B params, more manageable than 7B
        print(f"Using {model_name} - smaller model to avoid memory issues")

    print(f"Loading {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Load dataset
    print(f"Loading data from {args.data}")
    dataset = load_dataset('json', data_files={
        'train': args.data,
        'validation': args.data.replace('train', 'validation')
    })

    # Simple preprocessing
    def preprocess(examples):
        # Combine instruction, input, and output into single text
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"Task: {inst}\nText: {inp}\nEdit: {out}"
            texts.append(text)

        # Tokenize
        result = tokenizer(texts, padding=True, truncation=True, max_length=256)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(preprocess, batched=True)

    # Training arguments - very conservative
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=1,  # Just 1 epoch for testing
        per_device_train_batch_size=1,
        save_steps=20,
        logging_steps=5,
        eval_steps=20,
        evaluation_strategy="steps",
        warmup_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        report_to="none",
        max_steps=100 if args.test_mode else -1,  # Limit steps in test mode
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
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    print("✓ Done!")

if __name__ == "__main__":
    main()
