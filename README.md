# Editorial AI Assistant

A professional AI-powered editorial assistant that provides grammar correction, style improvements, and clarity enhancements for written content. Train on your own writing samples to create a personalized AI that writes in your unique style.

## Features

- **Multiple Model Sizes** - From 124M to 7B parameters, choose based on your hardware
- **Smart Hardware Detection** - Automatically detects GPU/CPU and recommends appropriate models
- **GPU Acceleration** - Full support for NVIDIA GPUs including RTX 5090/4090 series
- **Personal Style Training** - Train on your own emails and documents for a customized AI
- **Various Data Formats** - Supports CSV, Excel, Word documents, PDFs, emails (mbox/PST), and JSONL
- **Interactive Menu System** - User-friendly interface, no command-line expertise needed
- **Cross-Platform** - Works on Windows 11, Ubuntu/Linux, and macOS
- **Privacy-Focused** - Runs entirely offline on your local machine
- **Docker Container** - Easy deployment with all dependencies included

## Quick Start

### Prerequisites

#### For Ubuntu/Linux:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add yourself to docker group (to avoid using sudo)
sudo usermod -aG docker $USER
# Log out and back in, or run: newgrp docker

# For GPU support (optional but recommended)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### For Windows 11:
```powershell
# Open PowerShell as Administrator

# Install WSL2 first (if not already installed)
wsl --install

# Install Docker Desktop from:
# https://docs.docker.com/desktop/install/windows-install/

# OR use Docker in WSL2 (recommended):
# 1. Open WSL2 Ubuntu terminal
# 2. Follow the Ubuntu/Linux instructions above
```

### Running the Editorial AI Assistant

#### Ubuntu/Linux Command Line:
```bash
# Clone the repository
git clone https://github.com/kylefoxaustin/editorial-ai-assistant.git
cd editorial-ai-assistant

# Build the Docker image (first time only, takes 10-15 minutes)
docker build -t editorial-ai-assistant .

# Run with GPU support (if available)
docker run --rm -it --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    editorial-ai-assistant:latest

# Run without GPU (CPU only)
docker run --rm -it \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -e CUDA_VISIBLE_DEVICES="" \
    editorial-ai-assistant:latest
```

#### Windows 11 Command Line (PowerShell):
```powershell
# Clone the repository
git clone https://github.com/kylefoxaustin/editorial-ai-assistant.git
cd editorial-ai-assistant

# Build the Docker image (first time only, takes 10-15 minutes)
docker build -t editorial-ai-assistant .

# Run with GPU support
docker run --rm -it --gpus all `
    -v ${PWD}/models:/app/models `
    -v ${PWD}/data:/app/data `
    editorial-ai-assistant:latest

# Run without GPU (CPU only)
docker run --rm -it `
    -v ${PWD}/models:/app/models `
    -v ${PWD}/data:/app/data `
    -e CUDA_VISIBLE_DEVICES="" `
    editorial-ai-assistant:latest
```

## GPU Support

### RTX 5090/4090 and Modern GPU Support

For users with NVIDIA RTX 5090, 4090, or other modern GPUs, this project supports GPU acceleration through CUDA 12.8:

#### Requirements:
- NVIDIA Driver 525.60+ (check with `nvidia-smi`)
- Docker with nvidia-container-toolkit installed
- PyTorch with CUDA 12.8 support (included in container)

#### Enabling GPU Support:

If you encounter CUDA capability errors (especially on RTX 5090 with sm_120), the container automatically installs the correct PyTorch version:
```bash
# The container uses PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Performance Comparison:

| Hardware | Model Size | Training Time (approx) | Inference Speed |
|----------|------------|------------------------|-----------------|
| CPU (32 cores) | GPT-2 Base (124M) | 2-3 hours | 5-10 tokens/sec |
| CPU (32 cores) | GPT-2 Large (774M) | 20-30 hours | 1-3 tokens/sec |
| RTX 4090 | GPT-2 Large (774M) | 1-2 hours | 100+ tokens/sec |
| RTX 5090 | GPT-2 Large (774M) | 45-60 minutes | 150+ tokens/sec |
| RTX 5090 | Qwen 7B | 3-4 hours | 50+ tokens/sec |

## Training on Your Personal Writing Style

Transform the AI into your personal writing assistant by training on your own documents:

### Step 1: Prepare Your Data

Place your documents in `data/to_process/`:
- **Emails**: `.mbox` files (Gmail exports) or `.pst` files (Outlook)
- **Documents**: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- **Text files**: `.txt`, `.md`

### Step 2: Process Your Data

From the Docker container menu:
1. Select option 3: "Prepare training data"
2. Select option 3: "Process personal documents"
3. Choose the file type to process or "ALL"
4. The system will extract and prepare training examples

### Step 3: Filter Your Emails (Optional)

If processing email archives, you can filter to only your sent emails:
```bash
python3 scripts/filter_sender_emails.py \
    --input data/to_process/emails.mbox \
    --sender-emails "your@email.com,alternate@email.com" \
    --sender-name "YourName" \
    --output data/processed/
```

### Step 4: Train on Personal Data

1. Select option 1: "Train new model"
2. Choose your model size based on available hardware
3. Select option 2: "Personal style data"
4. Training begins automatically

The AI will learn your:
- Writing patterns and style
- Common phrases and expressions
- Sentence structure preferences
- Professional tone and voice

## Using the Assistant

Once the container starts, you'll see an interactive menu:
```
=========================================
Editorial AI Assistant Docker Container
=========================================
✓ GPU detected: NVIDIA GeForce RTX 5090
  GPU Memory: 32607MB

Available commands:
  1) Train new model
  2) Test existing model
  3) Prepare training data
  4) Bash shell
  5) Exit

Select option [1-5]:
```

### Training a Model

1. Select option `1` for "Train new model"
2. Choose a model size based on your hardware:
   - **No GPU/Low RAM**: GPT-2 Base (124M)
   - **Basic GPU (8GB)**: GPT-2 Large or XL
   - **Good GPU (12GB+)**: Phi-2 (2.7B)
   - **High-end GPU (24GB+)**: Qwen 2.5 7B

3. Set training parameters (or use defaults)
4. Training will begin automatically

### Testing Your Model

After training completes:
1. Select option `2` for "Test existing model"
2. Choose your trained model
3. Enter text for the AI to edit or improve
4. The AI will provide suggestions based on its training

## How to Use Your Trained Model

After training completes, you have multiple options for using your personalized AI:

### Option 1: Interactive Testing in Docker Container

The simplest way to start using your model immediately:
```bash
# Run the Docker container
docker run --rm -it --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    editorial-ai-assistant:latest

# Select option 2: "Test existing model"
# Choose your trained model
# Start entering prompts like:
#   - "Write an email about project updates"
#   - "Draft a message declining a meeting"
#   - "Explain our technical architecture"
```

### Option 2: Direct Python Script Usage

For more flexibility and integration into your workflow:
```bash
# Activate your virtual environment (if using one)
source editorial_venv/bin/activate

# Run the test script
python scripts/test_editorial.py
```

Or create a custom script (`my_ai_assistant.py`):
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load your trained model
model = GPT2LMHeadModel.from_pretrained('./models/editorial-ai-trained')
tokenizer = GPT2Tokenizer.from_pretrained('./models/editorial-ai-trained')

# Move to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.95
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
while True:
    topic = input("\nWhat do you need to write about? (or 'quit' to exit): ")
    if topic.lower() == 'quit':
        break

    # Generate in your style
    result = generate_text(f"Write about: {topic}\n\n")
    print(f"\nGenerated text:\n{result}")
```

### Option 3: Web Interface for Browser Access

Create a simple web interface (`app.py`):
```python
from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load model once at startup
model = GPT2LMHeadModel.from_pretrained('./models/editorial-ai-trained')
tokenizer = GPT2Tokenizer.from_pretrained('./models/editorial-ai-trained')

if torch.cuda.is_available():
    model = model.cuda()

@app.route('/')
def home():
    return '''
    <html>
        <head><title>Personal AI Assistant</title></head>
        <body style="font-family: Arial; max-width: 800px; margin: 50px auto;">
            <h1>Personal Writing Assistant</h1>
            <textarea id="prompt" rows="4" cols="80" placeholder="Enter your prompt here..."></textarea><br>
            <button onclick="generate()">Generate Text</button>
            <h3>Output:</h3>
            <div id="output" style="border: 1px solid #ccc; padding: 10px; min-height: 200px;"></div>

            <script>
                function generate() {
                    const prompt = document.getElementById('prompt').value;
                    fetch('/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt: prompt})
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('output').innerText = data.text;
                    });
                }
            </script>
        </body>
    </html>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    inputs = tokenizer(prompt, return_tensors='pt')

    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, temperature=0.8)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

Run with:
```bash
pip install flask
python app.py
# Open browser to http://localhost:5000
```

### Option 4: Export to Ollama for API Access (Recommended for Production)

Convert your model to GGUF format and serve via Ollama:
```bash
# Step 1: Export model to GGUF format
python scripts/export_to_gguf.py \
    --model-path ./models/editorial-ai-trained \
    --output-path ./models/personal-ai.gguf

# Step 2: Create Ollama Modelfile
cat > Modelfile << EOF
FROM ./models/personal-ai.gguf
PARAMETER temperature 0.8
PARAMETER top_p 0.95
SYSTEM "You are a helpful assistant that writes in the user's personal style."
EOF

# Step 3: Import to Ollama
ollama create personal-assistant -f Modelfile

# Step 4: Use via Ollama
ollama run personal-assistant

# Or use via API
curl http://localhost:11434/api/generate -d '{
  "model": "personal-assistant",
  "prompt": "Write an email about project delays"
}'
```

### Quick Start Examples

#### Email Generation
```python
prompt = "Write an email to the team about: new GPU server is ready for testing"
# AI generates email in your style
```

#### Document Editing
```python
prompt = "Rewrite this more concisely: [paste original text]"
# AI applies your writing style
```

#### Technical Writing
```python
prompt = "Explain our SOC architecture to a new engineer"
# AI explains in your voice
```

### Windows 11 Usage

For Windows 11 users (dual-boot or WSL2):

#### Native Windows (PowerShell)
```powershell
# Activate virtual environment
.\editorial_venv\Scripts\Activate

# Run test script
python scripts\test_editorial.py
```

#### WSL2
```bash
# Same as Linux instructions
source editorial_venv/bin/activate
python scripts/test_editorial.py
```

### Tips for Best Results

1. **Be specific with prompts**: "Write an email to management about Q3 delays" works better than "write something"

2. **Provide context**: "Reply to John's question about system requirements: [context]"

3. **Use your common phrases**: The model learned your style, so prompts with your typical language work best

4. **Adjust temperature**: Lower (0.5-0.7) for more focused, higher (0.8-1.0) for more creative

5. **Set appropriate length**: Shorter for emails (100-200 tokens), longer for documents (500+)

## Data Formats

### Editorial Training Data (CSV)
```csv
original,edited,type,explanation
"The report which was written by John was comprehensive","John's report was comprehensive","concision","Removed wordy construction"
"Its important to note that","It's important to note that","grammar","Fixed apostrophe"
```

### Personal Style Data
The system automatically extracts training examples from your documents:
- Emails: Extracts paragraphs from your sent messages
- Documents: Extracts text content and formatting
- Presentations: Extracts slide content and speaker notes

## Hardware Requirements

| Model | Parameters | VRAM Required | CPU RAM Required | Training Viable |
|-------|-----------|---------------|------------------|-----------------|
| GPT-2 Base | 124M | 2GB | 4GB | ✅ CPU/GPU |
| GPT-2 Medium | 355M | 4GB | 8GB | ✅ CPU/GPU |
| GPT-2 Large | 774M | 6GB | 12GB | ✅ GPU recommended |
| GPT-2 XL | 1.5B | 8GB | 16GB | ⚠️ GPU only |
| Phi-2 | 2.7B | 12GB | 24GB | ⚠️ GPU only |
| Qwen 2.5 7B | 7B | 24GB+ | 32GB+ | ❌ High-end GPU only |

## Project Structure
```
editorial-ai-assistant/
├── scripts/                      # Training and processing scripts
│   ├── train_model.py           # Main training script (with LoRA)
│   ├── train_simple.py          # Simple training script (full fine-tuning)
│   ├── test_model.py            # Model testing interface
│   ├── test_editorial.py        # Simple editorial testing
│   ├── prepare_data.py          # Basic data preparation
│   ├── prepare_data_enhanced.py # Multi-format data preparation
│   ├── prepare_universal_data.py # Universal document processor
│   └── filter_sender_emails.py  # Email filtering for personal training
├── data/
│   ├── raw/                     # Your input data files
│   ├── to_process/              # Documents to be processed
│   └── processed/               # Processed training data
├── models/                      # Trained models saved here
├── examples/                    # Sample edits and use cases
├── Dockerfile                   # Docker container definition
├── docker_startup.sh           # Interactive menu system
├── docker-compose.yml          # Docker compose configuration
├── requirements.txt            # Python dependencies
└── requirements-docker.txt     # Docker-specific dependencies
```

## Available Scripts

### Training Scripts
- **train_simple.py**: Full model fine-tuning (GPT-2 family)
- **train_model.py**: LoRA/PEFT training for large models

### Data Processing Scripts
- **prepare_universal_data.py**: Process any document type (PDF, DOCX, emails, etc.)
- **filter_sender_emails.py**: Extract only specific sender's emails from archives
- **prepare_data_enhanced.py**: Process editorial training data in multiple formats

### Testing Scripts
- **test_editorial.py**: Interactive testing interface
- **test_model.py**: Advanced model testing with metrics

## Troubleshooting

### "Permission denied" error on Linux:
```bash
# Make sure you're in the docker group
sudo usermod -aG docker $USER
newgrp docker
```

### GPU not detected:
```bash
# Check if nvidia-container-toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If that fails, install nvidia-container-toolkit:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### RTX 5090 "CUDA capability sm_120" error:
The container automatically installs PyTorch with CUDA 12.8 support. If issues persist:
```bash
# Inside container or virtual environment:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Out of memory errors:
- Choose a smaller model
- Reduce batch size
- Use CPU training (slower but works)
- Enable 8-bit quantization (for supported models)

### Windows-specific issues:
- Ensure WSL2 is installed and updated
- Docker Desktop must be running
- For GPU: Need NVIDIA GPU with WSL2 GPU support enabled

## Examples

### Training on Editorial Samples
```bash
# Start container
docker run --rm -it --gpus all -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data editorial-ai-assistant:latest

# Select: 1 (Train) → 1 (GPT-2 Base) → Enter defaults → Training begins
```

### Training on Personal Emails
```bash
# Place email exports in data/to_process/
# Start container and select: 3 (Prepare data) → 3 (Personal documents) → 1 (Emails)
# Then: 1 (Train) → Choose model → 2 (Personal style data)
```

### Testing Your Model
```bash
# After training, select: 2 (Test model) → 1 (Your trained model)
# Enter text like: "The report which was written by the committee was comprehensive"
# Get improved version: "The committee's report was comprehensive"
```

## License

MIT License - See [LICENSE](LICENSE) file for details

## Author & Maintainer

**Kyle Fox** - Austin, TX
GitHub: [@kylefoxaustin](https://github.com/kylefoxaustin)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution:
- Additional data format support
- Model optimization techniques
- UI improvements
- Documentation and examples
- Bug fixes and performance improvements

## Acknowledgments & Credits

This project builds upon several excellent open-source projects:

- **[Qwen 2.5](https://github.com/QwenLM/Qwen2.5)** by Alibaba Cloud - Base model for highest quality editing
- **[GPT-2](https://github.com/openai/gpt-2)** by OpenAI - Smaller models for CPU training
- **[Phi-2](https://huggingface.co/microsoft/phi-2)** by Microsoft - Medium-size model option
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Model framework
- **[PEFT](https://github.com/huggingface/peft)** - Parameter-efficient fine-tuning
- **[Ollama](https://ollama.ai)** - Model serving and deployment
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - GGUF conversion tools

Special thanks to the open-source AI community for making personalized AI accessible to everyone.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Citation

If you use this project in your research or work, please cite:
```bibtex
@software{editorial_ai_assistant,
  author = {Fox, Kyle},
  title = {Editorial AI Assistant: Personalized Writing Style Training},
  year = {2024},
  url = {https://github.com/kylefoxaustin/editorial-ai-assistant}
}
```
