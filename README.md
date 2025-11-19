# Editorial AI Assistant

A professional AI-powered editorial assistant that provides grammar correction, style improvements, and clarity enhancements for written content. Available as a Docker container for easy deployment on any system.

## Features

- **Multiple Model Sizes** - From 124M to 7B parameters, choose based on your hardware
- **Smart Hardware Detection** - Automatically detects GPU/CPU and recommends appropriate models
- **Various Data Formats** - Supports CSV, Excel, Word documents, and JSONL for training data
- **Interactive Menu System** - User-friendly interface, no command-line expertise needed
- **Cross-Platform** - Works on Windows 11, Ubuntu/Linux, and macOS
- **Privacy-Focused** - Runs entirely offline on your local machine

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

# Run with GPU support
docker run --rm -it --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    editorial-ai-assistant:latest

# Run without GPU (CPU only)
docker run --rm -it \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
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
    editorial-ai-assistant:latest
```

#### Windows 11 via WSL2 (Recommended):
```bash
# Open WSL2 Ubuntu terminal, then use the same commands as Ubuntu/Linux above
```

## Using the Assistant

Once the container starts, you'll see an interactive menu:
```
=========================================
Editorial AI Assistant Docker Container
=========================================
✓ GPU detected: NVIDIA GeForce RTX 4080
  GPU Memory: 16384MB

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

### Preparing Your Own Training Data

Your training data should contain pairs of original and edited text. Supported formats:

#### CSV Format (recommended):
Create a file `data/raw/my_edits.csv`:
```csv
original,edited,type,explanation
"The report which was written by John was comprehensive","John's report was comprehensive","concision","Removed wordy construction"
"Its important to note that","It's important to note that","grammar","Fixed apostrophe"
```

#### Excel Format:
Same structure as CSV, saved as `.xlsx`

#### Word Document:
Alternating paragraphs of original and edited text

#### JSONL Format:
```json
{"instruction": "Fix grammar", "input": "Original text", "output": "Edited text"}
{"instruction": "Make concise", "input": "Another original", "output": "Another edit"}
```

To process your data:
1. Place your file in `data/raw/`
2. Select option `3` from main menu
3. Choose your file type
4. Enter filename
5. Data will be processed automatically

## Hardware Requirements

| Model | Parameters | VRAM Required | CPU Training | Quality |
|-------|-----------|---------------|--------------|---------|
| GPT-2 Base | 124M | 2GB | ✅ Fast | Basic |
| GPT-2 Medium | 355M | 4GB | ✅ Slow | Good |
| GPT-2 Large | 774M | 6GB | ⚠️ Very Slow | Good |
| GPT-2 XL | 1.5B | 8GB | ❌ Too Slow | Better |
| Phi-2 | 2.7B | 12GB | ❌ Too Slow | Very Good |
| Qwen 2.5 7B | 7B | 24GB+ | ❌ Impractical | Best |

## Example Use Cases

- **Professional Writers**: Improve clarity and concision
- **Academic Writing**: Fix grammar and enhance style
- **Business Communications**: Remove jargon and improve readability
- **Content Creators**: Polish blog posts and articles
- **Students**: Learn better writing patterns

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

### Out of memory errors:
- Choose a smaller model
- Reduce batch size
- Use CPU training (slower but works)

### Windows-specific issues:
- Make sure WSL2 is installed and updated
- Docker Desktop must be running
- For GPU: Need NVIDIA GPU with WSL2 GPU support enabled

## Project Structure
```
editorial-ai-assistant/
├── scripts/                      # Training and inference scripts
│   ├── train_model.py           # Main training script
│   ├── test_model.py            # Model testing
│   ├── prepare_data.py          # Basic data preparation
│   └── prepare_data_enhanced.py # Advanced multi-format data prep
├── data/
│   ├── raw/                     # Your input data files
│   └── processed/                # Processed training data
├── models/                       # Trained models saved here
├── examples/                     # Sample edits and use cases
├── Dockerfile                    # Docker container definition
├── docker_startup.sh            # Interactive menu system
├── requirements.txt             # Python dependencies
└── requirements-docker.txt      # Docker-specific dependencies
```

## License

MIT License - See [LICENSE](LICENSE) file for details

## Author & Maintainer

**Kyle Fox** - Austin, TX  
GitHub: [@kylefoxaustin](https://github.com/kylefoxaustin)

## Acknowledgments & Credits

This project builds upon several excellent open-source projects:

- **[Qwen 2.5](https://github.com/QwenLM/Qwen2.5)** by Alibaba Cloud - Base model for highest quality editing
- **[GPT-2](https://github.com/openai/gpt-2)** by OpenAI - Smaller models for CPU training
- **[Phi-2](https://huggingface.co/microsoft/phi-2)** by Microsoft - Medium-size model option
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Model framework
- **[PEFT](https://github.com/huggingface/peft)** - Parameter-efficient fine-tuning
- **[Ollama](https://ollama.ai)** - Model serving and deployment
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - GGUF conversion tools

Special thanks to the open-source AI community for making projects like this possible.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
