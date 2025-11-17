#!/usr/bin/env python3
"""
RTX 5090 Compatible Training Script - CPU Setup Method
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import argparse

def setup_model_and_tokenizer(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Load model on CPU first to avoid CUDA issues"""

    print(f"Loading {model_name}...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model on CPU first
    print("Loading model on CPU to avoid CUDA kernel issues...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU first
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")

    return model, tokenizer

def setup_lora_and_move_to_gpu(model):
    """Setup LoRA on CPU then move to GPU"""

    print("Setting up LoRA configuration...")

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Small rank to save memory
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Just attention for now
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA while on CPU
    print("Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Now move to GPU
    print("Moving model to GPU...")
    model = model.cuda()

    # Enable gradient checkpointing after moving to GPU
    model.gradient_checkpointing_enable()

    print("✓ Model ready on GPU")

    return model

def preprocess_function(examples, tokenizer, max_length=256):
    """Format examples for training"""

    prompts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        # Simplified prompt format
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        prompts.append(prompt)

    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir="./editorial-ai-model",
    num_epochs=3,
    batch_size=1,
    learning_rate=3e-4
):
    """Train with minimal settings to avoid issues"""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,  # Avoid model reloading issues
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=0.3,  # More aggressive gradient clipping
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        ),
    )

    print("Starting training...")
    try:
        trainer.train()

        print(f"Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

    except RuntimeError as e:
        print(f"Training error: {e}")
        print("Attempting to save current state anyway...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Editorial AI")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--data", default="data/processed/train.jsonl", help="Training data")
    parser.add_argument("--output", default="./editorial-ai-model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected!")
        return
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory")

    # Load model on CPU
    model, tokenizer = setup_model_and_tokenizer(args.model)

    # Setup LoRA and move to GPU
    model = setup_lora_and_move_to_gpu(model)

    # Load datasets
    print(f"Loading dataset from {args.data}")
    dataset = load_dataset('json', data_files={
        'train': args.data,
        'validation': args.data.replace('train', 'validation')
    })

    print(f"Dataset size: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    # Preprocess
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length=256),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Train
    trainer = train_model(
        model,
        tokenizer,
        tokenized_dataset["train"],
        tokenized_dataset["validation"],
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("✓ Training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nNext steps:")
    print("1. Test: python scripts/test_model.py --model ./editorial-ai-model")
    print("2. Export: python scripts/export_ollama.py")

if __name__ == "__main__":
    main()
