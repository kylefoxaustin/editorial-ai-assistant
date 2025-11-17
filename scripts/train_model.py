#!/usr/bin/env python3
"""
Fine-tuning Script for Editorial AI Assistant
Uses PEFT/LoRA for efficient training on consumer GPUs
"""

import os
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
    TaskType,
    prepare_model_for_kbit_training
)
import argparse

def setup_model_and_tokenizer(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_8bit=True):
    """Load model and tokenizer with memory optimizations"""
    
    print(f"Loading {model_name}...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Model with quantization for memory efficiency
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
    
    return model, tokenizer

def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.1):
    """Configure LoRA for efficient fine-tuning"""
    
    lora_config = LoraConfig(
        r=r,  # Rank
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def preprocess_function(examples, tokenizer, max_length=512):
    """Format examples for training"""
    
    # Create prompt template
    prompts = []
    for instruction, input_text, output in zip(
        examples["instruction"], 
        examples["input"], 
        examples["output"]
    ):
        prompt = f"""<|im_start|>system
You are a professional editor providing high-quality editorial feedback.<|im_end|>
<|im_start|>user
{instruction}

Text: {input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        prompts.append(prompt)
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Set labels (same as input_ids for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir="./editorial-ai-model",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
):
    """Train the model"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=-1,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        logging_first_step=True,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # Change to "wandb" if you want logging
        remove_unused_columns=False,
        gradient_checkpointing=True,
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
    trainer.train()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Editorial AI")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--data", default="data/processed/train.jsonl", help="Training data")
    parser.add_argument("--output", default="./editorial-ai-model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  Warning: No GPU detected. Training will be very slow!")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ Using GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model, args.use_8bit)
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load datasets
    print(f"Loading dataset from {args.data}")
    dataset = load_dataset('json', data_files={
        'train': args.data,
        'validation': args.data.replace('train', 'validation')
    })
    
    # Preprocess
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
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
    print("1. Test the model with test_model.py")
    print("2. Export to Ollama with export_ollama.py")

if __name__ == "__main__":
    main()
