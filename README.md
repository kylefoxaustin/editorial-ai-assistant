# Editorial AI Assistant

A fine-tuned Qwen 2.5 7B language model specialized for professional editorial assistance, providing grammar correction, style improvements, and clarity enhancements.

## Features
- Grammar and punctuation correction
- Concision and clarity improvements  
- Active/passive voice conversion
- Business jargon simplification
- Style and tone adjustments

## Project Structure
```
editorial-ai-assistant/
├── scripts/
│   ├── train_model.py          # Main training script
│   ├── test_model.py            # Model testing
│   ├── prepare_data.py          # Data preprocessing
│   └── export_ollama.py         # Ollama export utility
├── data/
│   ├── raw/                     # Original training data (CSV)
│   └── processed/                # Processed JSONL files
└── models/                       # Model checkpoints (not in repo)
```

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM for inference, 40GB+ for training)
- 20GB+ disk space for models

### Installation
```bash
# Clone repository
git clone https://github.com/kylefoxaustin/editorial-ai-assistant.git
cd editorial-ai-assistant

# Create virtual environment
python -m venv editorial_env
source editorial_env/bin/activate  # On Windows: editorial_env\Scripts\activate

# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes
```

## Training

### Prepare Data
```bash
python scripts/prepare_data.py
```

### Train Model
```bash
python scripts/train_model.py \
    --model 'Qwen/Qwen2.5-7B-Instruct' \
    --data 'data/processed/train.jsonl' \
    --epochs 3 \
    --batch-size 2 \
    --use-8bit
```

Training takes approximately 1-2 hours on an RTX 8000 or similar GPU.

## Usage

### Test the Model
```bash
python scripts/test_model.py --model ./editorial-ai-model
```

### Export to Ollama
```bash
python scripts/export_ollama.py --model ./editorial-ai-model
ollama run editorial-assistant
```

## Pre-trained Models

Due to size constraints (15GB+), trained models are not included in this repository.

Options:
1. Train your own using the scripts provided
2. Contact the maintainer for pre-trained model access
3. Use the cloud-hosted version (coming soon)

## Example Results

Input: "The report which was written by John was very comprehensive."
Output: "The report written by John was very comprehensive."

Input: "We need to leverage our core competencies to synergize our value proposition."  
Output: "We should focus on our strengths to improve our customer offering."

## Hardware Requirements

- **Training**: 40GB+ VRAM (RTX 8000, A100, or similar)
- **Inference**: 8GB+ VRAM (RTX 3070 or better)
- **CPU Only**: Possible but slow (not recommended)

## Known Issues

- RTX 5090 currently incompatible due to PyTorch CUDA kernel support
- Use PyTorch nightly builds for bleeding-edge GPU support

## License

MIT - See LICENSE file for details

## Acknowledgments

- Based on Qwen 2.5 7B by Alibaba Cloud
- Training methodology inspired by LoRA/PEFT papers
- Special thanks to the editorial team for providing training examples
