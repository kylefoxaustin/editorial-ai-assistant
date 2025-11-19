#!/usr/bin/env python3
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading Editorial AI model...")
model = GPT2LMHeadModel.from_pretrained("./models/editorial-ai-trained")
tokenizer = GPT2Tokenizer.from_pretrained("./models/editorial-ai-trained")
tokenizer.pad_token = tokenizer.eos_token

print("\nEditorial AI Ready! (type 'quit' to exit)\n")

while True:
    text = input("Enter text to edit: ")
    if text.lower() in ['quit', 'exit']:
        break
    
    prompt = f"Edit: {text}\nResult:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.encode("\n")[0]  # Stop at newline
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    edited = result.split("Result:")[-1].split("\n")[0].strip()
    
    print(f"Edited: {edited}\n")
