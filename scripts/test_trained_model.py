from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the model
print("Loading model...")
base_model_name = "Qwen/Qwen2.5-7B-Instruct"
model_path = "./editorial-ai-model-full"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)
model = PeftModel.from_pretrained(model, model_path)
model.eval()

# Test examples
test_cases = [
    "Fix grammar: The report which was written by John was very comprehensive.",
    "Make concise: We need to leverage our core competencies to synergize our value proposition.",
    "Convert to active voice: The decision was made by the committee."
]

print("\n" + "="*60)
print("Testing Editorial AI Model")
print("="*60)

for test in test_cases:
    prompt = f"<|im_start|>system\nYou are a professional editor.<|im_end|>\n<|im_start|>user\n{test}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|im_start|>assistant")[-1].strip()
    
    print(f"\nInput: {test}")
    print(f"Output: {response[:200]}")
    print("-"*40)

print("\n✓ Model is working!")
