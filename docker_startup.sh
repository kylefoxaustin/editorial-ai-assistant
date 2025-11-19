#!/bin/bash

echo "========================================="
echo "Editorial AI Assistant Docker Container"
echo "========================================="

# Check for GPU
if nvidia-smi &> /dev/null 2>&1; then
    echo "✓ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    export CUDA_AVAILABLE="true"
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    echo "  GPU Memory: ${GPU_MEM}MB"
else
    echo "⚠ No GPU detected - using CPU"
    echo "  Training will be slower but will still work"
    export CUDA_AVAILABLE="false"
fi

# Interactive menu if no command specified
if [ $# -eq 0 ]; then
    echo ""
    echo "Available commands:"
    echo "  1) Train new model"
    echo "  2) Test existing model"
    echo "  3) Prepare training data"
    echo "  4) Bash shell"
    echo "  5) Exit"
    echo ""
    read -p "Select option [1-5]: " choice

    case $choice in
        1)
            echo ""
            echo "Select model size based on your hardware:"
            echo ""
            echo "  Small Models (CPU or low-end GPU):"
            echo "    1) GPT-2 Base (124M params) - 2GB VRAM - Good for testing"
            echo "    2) GPT-2 Medium (355M params) - 4GB VRAM"
            echo ""
            echo "  Medium Models (decent GPU):"
            echo "    3) GPT-2 Large (774M params) - 6GB VRAM"
            echo "    4) GPT-2 XL (1.5B params) - 8GB VRAM"
            echo ""
            echo "  Large Models (high-end GPU):"
            echo "    5) Phi-2 (2.7B params) - 12GB VRAM - Good quality"
            echo "    6) Qwen 2.5 7B (7B params) - 24GB+ VRAM - Best quality"
            echo ""

            if [ "$CUDA_AVAILABLE" == "false" ]; then
                echo "  ⚠ CPU detected - recommend options 1 or 2"
            elif [ "$GPU_MEM" -lt 8000 ]; then
                echo "  ⚠ GPU has ${GPU_MEM}MB - recommend options 1-2"
            elif [ "$GPU_MEM" -lt 12000 ]; then
                echo "  ✓ GPU has ${GPU_MEM}MB - options 1-4 will work"
            elif [ "$GPU_MEM" -lt 24000 ]; then
                echo "  ✓ GPU has ${GPU_MEM}MB - options 1-5 will work"
            else
                echo "  ✓ GPU has ${GPU_MEM}MB - all options available!"
            fi

            echo ""
            read -p "Select model [1-6]: " model_choice

            case $model_choice in
                1) MODEL="gpt2"; VRAM_NEEDED="2GB" ;;
                2) MODEL="gpt2-medium"; VRAM_NEEDED="4GB" ;;
                3) MODEL="gpt2-large"; VRAM_NEEDED="6GB" ;;
                4) MODEL="gpt2-xl"; VRAM_NEEDED="8GB" ;;
                5) MODEL="microsoft/phi-2"; VRAM_NEEDED="12GB" ;;
                6) MODEL="Qwen/Qwen2.5-7B-Instruct"; VRAM_NEEDED="24GB" ;;
                *) echo "Invalid choice"; exit 1 ;;
            esac

            echo ""
            echo "Selected: $MODEL (requires ~$VRAM_NEEDED)"
            echo ""
            read -p "Number of training epochs (1-5, default 3): " epochs
            epochs=${epochs:-3}

            read -p "Batch size (1-8, default 2): " batch_size
            batch_size=${batch_size:-2}

            echo ""
            echo "Starting training with:"
            echo "  Model: $MODEL"
            echo "  Epochs: $epochs"
            echo "  Batch size: $batch_size"
            echo ""

            # Use 8-bit quantization for large models on GPU
            if [ "$CUDA_AVAILABLE" == "true" ] && [ "$model_choice" -ge 5 ]; then
                python3 scripts/train_model.py \
                    --model "$MODEL" \
                    --epochs $epochs \
                    --batch-size $batch_size \
                    --use-8bit
            else
                python3 scripts/train_model.py \
                    --model "$MODEL" \
                    --epochs $epochs \
                    --batch-size $batch_size
            fi
            ;;

        2)
            echo ""
            echo "Testing model..."
            if [ -d "./models/editorial-ai-model" ]; then
                python3 scripts/test_model.py --model ./models/editorial-ai-model
            else
                echo "No trained model found. Train a model first!"
            fi
            ;;

        3)
            echo ""
            echo "Data Preparation Options:"
            echo ""
            echo "  1) Process CSV file (columns: original, edited, type, explanation)"
            echo "  2) Process Excel file (.xlsx with same columns as CSV)"
            echo "  3) Process Word document (.docx with alternating paragraphs)"
            echo "  4) Process JSONL file (JSON Lines format)"
            echo "  5) Show data format examples"
            echo ""
            read -p "Select option [1-5]: " data_choice

            case $data_choice in
                1)
                    echo "Place your CSV file in data/raw/ directory"
                    read -p "Enter filename (or press Enter for sample_editorial_data.csv): " filename
                    filename=${filename:-sample_editorial_data.csv}
                    python3 scripts/prepare_data_enhanced.py --input "data/raw/$filename" --input-type csv
                    ;;
                2)
                    echo "Place your Excel file in data/raw/ directory"
                    read -p "Enter filename: " filename
                    python3 scripts/prepare_data_enhanced.py --input "data/raw/$filename" --input-type xlsx
                    ;;
                3)
                    echo "Place your Word document in data/raw/ directory"
                    read -p "Enter filename: " filename
                    python3 scripts/prepare_data_enhanced.py --input "data/raw/$filename" --input-type docx
                    ;;
                4)
                    echo "Place your JSONL file in data/raw/ directory"
                    read -p "Enter filename: " filename
                    python3 scripts/prepare_data_enhanced.py --input "data/raw/$filename" --input-type jsonl
                    ;;
                5)
                    python3 scripts/prepare_data_enhanced.py --show-examples
                    ;;
                *)
                    echo "Invalid choice"
                    ;;
            esac
            ;;

        4)
            echo "Entering bash shell..."
            echo "Type 'exit' to return to menu"
            /bin/bash
            ;;

        5)
            echo "Exiting..."
            exit 0
            ;;

        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
else
    # Execute passed command directly
    exec "$@"
fi

# After completing any action, ask if user wants to continue
echo ""
read -p "Press Enter to return to menu, or Ctrl+C to exit..."
exec $0  # Re-run this script to show menu again
