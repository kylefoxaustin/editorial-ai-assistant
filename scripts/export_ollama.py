#!/usr/bin/env python3
"""
Export Editorial AI Model to Ollama
Converts fine-tuned model to GGUF format for Ollama
"""

import os
import subprocess
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_weights(model_path, base_model, output_path):
    """Merge LoRA weights with base model"""
    
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path

def convert_to_gguf(model_path, output_file, quantization="Q4_K_M"):
    """Convert model to GGUF format"""
    
    print(f"Converting to GGUF format with {quantization} quantization...")
    
    # Check if llama.cpp is installed
    llama_cpp_path = Path.home() / "llama.cpp"
    if not llama_cpp_path.exists():
        print("Installing llama.cpp...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_cpp_path)
        ], check=True)
        
        # Build llama.cpp
        subprocess.run(["make", "-C", str(llama_cpp_path)], check=True)
    
    convert_script = llama_cpp_path / "convert.py"
    quantize_exe = llama_cpp_path / "quantize"
    
    # Convert to GGUF F16
    temp_gguf = Path(output_file).with_suffix('.f16.gguf')
    
    print("Converting to GGUF...")
    subprocess.run([
        "python", str(convert_script),
        str(model_path),
        "--outtype", "f16",
        "--outfile", str(temp_gguf)
    ], check=True)
    
    # Quantize
    print(f"Quantizing to {quantization}...")
    subprocess.run([
        str(quantize_exe),
        str(temp_gguf),
        str(output_file),
        quantization
    ], check=True)
    
    # Clean up temp file
    temp_gguf.unlink()
    
    print(f"✓ Model converted to {output_file}")
    return output_file

def create_modelfile(model_name, gguf_path, system_prompt=None):
    """Create Ollama Modelfile"""
    
    if system_prompt is None:
        system_prompt = """You are a professional editorial assistant trained to help improve written content. 
You provide clear, constructive feedback on grammar, style, clarity, and structure. 
You maintain the author's voice while enhancing readability and professionalism."""
    
    modelfile_content = f"""# Editorial AI Assistant
# Based on Qwen 2.5 7B, fine-tuned for editorial work

FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """{system_prompt}"""

LICENSE """
Custom Editorial AI Model
Fine-tuned for professional editorial assistance.
"Based on Qwen 2.5 7B model."
"""
"""
    
    modelfile_path = Path(f"{model_name}.Modelfile")
    modelfile_path.write_text(modelfile_content)
    
    print(f"✓ Created Modelfile: {modelfile_path}")
    return modelfile_path

def import_to_ollama(model_name, modelfile_path):
    """Import model into Ollama"""
    
    print(f"Creating Ollama model: {model_name}")
    
    try:
        # Create the model in Ollama
        subprocess.run([
            "ollama", "create",
            model_name,
            "-f", str(modelfile_path)
        ], check=True)
        
        print(f"✓ Model created in Ollama as '{model_name}'")
        print(f"\nYou can now use it with:")
        print(f"  ollama run {model_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to create Ollama model: {e}")
        print("Make sure Ollama is installed and running")
        print("You can manually create it with:")
        print(f"  ollama create {model_name} -f {modelfile_path}")

def main():
    parser = argparse.ArgumentParser(description="Export model to Ollama")
    parser.add_argument("--model", default="./editorial-ai-model", help="Fine-tuned model path")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model (if using LoRA)")
    parser.add_argument("--output", default="editorial-assistant.gguf", help="Output GGUF file")
    parser.add_argument("--quantization", default="Q4_K_M", 
                       choices=["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0"],
                       help="Quantization type")
    parser.add_argument("--name", default="editorial-assistant", help="Ollama model name")
    parser.add_argument("--merge-only", action="store_true", help="Only merge LoRA, don't convert")
    parser.add_argument("--no-ollama", action="store_true", help="Don't import to Ollama")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    # Check if it's a LoRA model
    if (model_path / "adapter_config.json").exists():
        print("Detected LoRA adapter, merging with base model...")
        merged_path = Path("./merged-model")
        merge_lora_weights(model_path, args.base_model, merged_path)
        model_path = merged_path
    
    if args.merge_only:
        print(f"✓ Merged model saved to {model_path}")
        return
    
    # Convert to GGUF
    gguf_path = convert_to_gguf(model_path, args.output, args.quantization)
    
    if not args.no_ollama:
        # Create Modelfile
        modelfile = create_modelfile(args.name, gguf_path)
        
        # Import to Ollama
        import_to_ollama(args.name, modelfile)
    
    print("\n✅ Export complete!")
    print(f"GGUF file: {gguf_path}")
    if not args.no_ollama:
        print(f"Ollama model: {args.name}")

if __name__ == "__main__":
    main()
