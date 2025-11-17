#!/usr/bin/env python3
"""
Testing Script for Editorial AI Assistant
Interactive testing and batch evaluation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from pathlib import Path

class EditorialAssistant:
    def __init__(self, model_path="./editorial-ai-model", base_model=None):
        """Initialize the editorial assistant"""
        
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if it's a LoRA model or full model
        config_path = Path(model_path) / "adapter_config.json"
        
        if config_path.exists() and base_model:
            # Load base model + LoRA adapter
            print(f"Loading base model: {base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            # Load full fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        
        self.model.eval()
        print("✓ Model loaded and ready!")
        
    def edit_text(self, text, instruction=None, max_length=512, temperature=0.7):
        """Edit a piece of text"""
        
        if instruction is None:
            instruction = "Edit this text for clarity and style"
        
        # Create prompt
        prompt = f"""<|im_start|>system
You are a professional editor providing high-quality editorial feedback.<|im_end|>
<|im_start|>user
{instruction}

Text: {text}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
        
        return response.strip()
    
    def interactive_mode(self):
        """Run interactive editing session"""
        
        print("\n" + "="*60)
        print("Editorial AI Assistant - Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /clear    - Clear the conversation")
        print("  /exit     - Exit the program")
        print("  /temp X   - Set temperature (0.1-2.0)")
        print("="*60)
        print("\nEnter text to edit (or /exit to quit):\n")
        
        temperature = 0.7
        
        while True:
            # Get input
            print("\n" + "-"*40)
            text = input("Original text: ").strip()
            
            # Check for commands
            if text.startswith("/"):
                if text == "/exit":
                    print("Goodbye!")
                    break
                elif text == "/help":
                    print("Enter any text to get editorial feedback.")
                    print("You can also specify custom instructions.")
                    continue
                elif text.startswith("/temp"):
                    try:
                        temperature = float(text.split()[1])
                        print(f"Temperature set to {temperature}")
                    except:
                        print("Usage: /temp 0.7")
                    continue
                elif text == "/clear":
                    print("\033[2J\033[H")  # Clear screen
                    continue
                else:
                    print(f"Unknown command: {text}")
                    continue
            
            if not text:
                continue
            
            # Get custom instruction if desired
            instruction = input("Instruction (or Enter for default): ").strip()
            if not instruction:
                instruction = None
            
            # Process
            print("\n📝 Editorial feedback:")
            print("-"*40)
            
            try:
                edited = self.edit_text(text, instruction, temperature=temperature)
                print(edited)
            except Exception as e:
                print(f"Error: {e}")
    
    def test_examples(self):
        """Test on standard examples"""
        
        test_cases = [
            {
                "text": "The report which was written by John was very comprehensive and it contained alot of good informations.",
                "instruction": "Fix grammar and improve concision"
            },
            {
                "text": "We need to leverage our core competencies to synergize our value proposition.",
                "instruction": "Simplify corporate jargon"
            },
            {
                "text": "The defendant was found to be not guilty by the jury after a long deliberation process that lasted several hours.",
                "instruction": "Make more concise using active voice"
            }
        ]
        
        print("\n" + "="*60)
        print("Testing Editorial AI on Sample Texts")
        print("="*60)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n[Test {i}]")
            print(f"Original: {case['text']}")
            print(f"Instruction: {case['instruction']}")
            print(f"Edited: {self.edit_text(case['text'], case['instruction'])}")
            print("-"*40)

def main():
    parser = argparse.ArgumentParser(description="Test Editorial AI")
    parser.add_argument("--model", default="./editorial-ai-model", help="Model path")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model (if using LoRA)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--test", action="store_true", help="Run test examples")
    parser.add_argument("--text", type=str, help="Single text to edit")
    parser.add_argument("--instruction", type=str, help="Custom instruction")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = EditorialAssistant(args.model, args.base_model)
    
    if args.interactive:
        assistant.interactive_mode()
    elif args.test:
        assistant.test_examples()
    elif args.text:
        edited = assistant.edit_text(args.text, args.instruction)
        print(f"\nOriginal: {args.text}")
        print(f"Edited: {edited}")
    else:
        # Default to interactive mode
        assistant.interactive_mode()

if __name__ == "__main__":
    main()
