#!/usr/bin/env python3
"""Simple training script that actually works"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model to use")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading training data...")
    with open("data/processed/train.jsonl", "r") as f:
        train_texts = []
        for line in f:
            data = json.loads(line)
            # Format: "Edit: [input] Result: [output]"
            text = f"Edit: {data['input']}\nResult: {data['output']}"
            train_texts.append(text)
    
    print(f"Found {len(train_texts)} training examples")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=256)
    
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": train_texts})
    tokenized = dataset.map(tokenize_function, batched=True)
    
    # Data collator that handles labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 doesn't use masked language modeling
    )
    
    # Training args
    training_args = TrainingArguments(
        output_dir="./models/editorial-ai-trained",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=50,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=10,
        warmup_steps=10,
        logging_dir='./logs',
        report_to="none",  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save
    print("Saving model...")
    model.save_pretrained("./models/editorial-ai-trained")
    tokenizer.save_pretrained("./models/editorial-ai-trained")
    print("✓ Training complete! Model saved to ./models/editorial-ai-trained")

if __name__ == "__main__":
    main()
