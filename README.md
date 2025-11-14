# Editorial AI Assistant

An AI assistant specifically trained to help with editorial work, built by fine-tuning Qwen 2.5 7B on editorial correspondence and feedback.

## Features

- 📝 Provides editorial feedback on manuscripts
- ✍️ Suggests line edits for clarity and style
- 💬 Answers questions about editorial decisions
- 🎯 Matches your editorial voice and preferences
- 🔒 Runs 100% locally on your hardware

## Hardware Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on RTX 5090)
- 32GB+ system RAM
- 100GB+ free disk space
- Ubuntu 22.04+ or similar Linux distribution

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kylefoxaustin/editorial-ai-assistant.git
cd editorial-ai-assistant

# Run setup (creates virtual environment and installs dependencies)
./setup.sh

# Activate environment
workon editorial_ai

# Prepare your training data
python scripts/prepare_data.py

# Train the model
python scripts/train_model.py

# Test the model
python scripts/test_model.py

# Launch web interface
python web_interface/app.py
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Data Preparation](docs/data_preparation.md)
- [Training Guide](docs/training_guide.md)
- [Deployment Guide](docs/deployment.md)

## Project Structure

```
editorial-ai-assistant/
├── docs/              # Documentation
├── scripts/           # Training and utility scripts
├── src/              # Source code
├── examples/         # Example data and usage
├── web_interface/    # Gradio web UI
└── tests/            # Unit tests
```

## Training Your Own Model

1. **Gather Data**: Collect editorial feedback, correspondence, and examples
2. **Prepare Data**: Run `prepare_data.py` to format for training
3. **Train**: Run `train_model.py` (takes 24-48 hours on RTX 5090)
4. **Deploy**: Launch the web interface

See [docs/training_guide.md](docs/training_guide.md) for details.

## Privacy & Security

- All training happens locally on your machine
- No data is sent to external servers
- Model stays on your hardware
- Training data should be anonymized

## Performance

**On RTX 5090:**
- Training time: ~36 hours (500 examples)
- Inference speed: ~20-40 tokens/second
- Memory usage: ~8-12GB VRAM (4-bit quantization)

## Contributing

This is a personal project, but suggestions and improvements are welcome! Open an issue or PR.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Base model: [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Fast training: [Unsloth](https://github.com/unslothai/unsloth)
- Inspired by work with Claude AI for project planning

## Author

[@kylefoxaustin](https://github.com/kylefoxaustin) - Austin, TX

Built alongside nanochat training experiments on RTX 5090 and RTX 8000.
