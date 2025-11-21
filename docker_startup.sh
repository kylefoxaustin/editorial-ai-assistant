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
            fi

            echo ""
            read -p "Select model [1-6]: " model_choice

            case $model_choice in
                1) MODEL="gpt2"; SCRIPT="train_simple.py" ;;
                2) MODEL="gpt2-medium"; SCRIPT="train_simple.py" ;;
                3) MODEL="gpt2-large"; SCRIPT="train_simple.py" ;;
                4) MODEL="gpt2-xl"; SCRIPT="train_simple.py" ;;
                5) MODEL="microsoft/phi-2"; SCRIPT="train_model.py" ;;
                6) MODEL="Qwen/Qwen2.5-7B-Instruct"; SCRIPT="train_model.py" ;;
                *) echo "Invalid choice"; exit 1 ;;
            esac

            echo ""
            read -p "Number of training epochs (1-5, default 3): " epochs
            epochs=${epochs:-3}

            read -p "Batch size (1-8, default 2): " batch_size
            batch_size=${batch_size:-2}

            echo ""
            echo "Select training data:"
            echo "  1) Standard editorial examples (default)"
            echo "  2) Personal style data (if processed)"
            echo ""
            read -p "Select data source [1-2]: " data_choice
            
            if [ "$data_choice" == "2" ]; then
                if [ -f "data/processed/personal_style_train.jsonl" ]; then
                    DATA_FILE="data/processed/personal_style_train.jsonl"
                    echo "Using personal style training data"
                else
                    echo "Personal data not found. Using standard data."
                    DATA_FILE="data/processed/train.jsonl"
                fi
            else
                DATA_FILE="data/processed/train.jsonl"
            fi

            echo ""
            echo "Starting training with:"
            echo "  Model: $MODEL"
            echo "  Epochs: $epochs"
            echo "  Batch size: $batch_size"
            echo "  Data: $DATA_FILE"
            echo ""

            # Use appropriate script based on model
            if [ "$SCRIPT" == "train_simple.py" ]; then
                # GPT-2 models use the simple script
                python3 scripts/train_simple.py \
                    --model "$MODEL" \
                    --epochs $epochs \
                    --batch-size $batch_size
            else
                # Larger models use the full script with LoRA
                if [ "$CUDA_AVAILABLE" == "true" ]; then
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
            fi
            ;;

        2)
            echo ""
            echo "Available models:"
            if [ -d "./models/editorial-ai-trained" ]; then
                echo "  1) editorial-ai-trained (GPT-2 fine-tuned)"
            fi
            if [ -d "./models/editorial-ai-model" ]; then
                echo "  2) editorial-ai-model (base model)"
            fi
            if [ -d "./models/personal-style-model" ]; then
                echo "  3) personal-style-model (your personalized model)"
            fi
            echo "  4) Enter custom model path"
            echo ""
            read -p "Select model [1-4]: " test_choice
            
            case $test_choice in
                1) MODEL_PATH="./models/editorial-ai-trained" ;;
                2) MODEL_PATH="./models/editorial-ai-model" ;;
                3) MODEL_PATH="./models/personal-style-model" ;;
                4) 
                    read -p "Enter model path: " MODEL_PATH ;;
                *) MODEL_PATH="./models/editorial-ai-trained" ;;
            esac
            
            # Use the simple test script for GPT-2 models
            if [[ "$MODEL_PATH" == *"gpt2"* ]] || [[ "$MODEL_PATH" == *"editorial-ai-trained"* ]]; then
                python3 scripts/test_editorial.py
            else
                python3 scripts/test_model.py --model "$MODEL_PATH"
            fi
            ;;

        3)
            echo ""
            echo "Data Preparation Options:"
            echo "  1) Use existing sample editorial data"
            echo "  2) Process new CSV file with editorial examples"
            echo "  3) Process personal documents (emails, PDFs, etc.)"
            echo "  4) Show data format examples"
            echo ""
            read -p "Select option [1-4]: " data_choice
            
            case $data_choice in
                1)
                    python3 scripts/prepare_data.py
                    echo "✓ Sample data ready for training"
                    ;;
                2)
                    echo "Place your CSV file in data/raw/"
                    echo "Format: columns should be 'original', 'edited', 'type', 'explanation'"
                    read -p "Enter CSV filename: " filename
                    python3 scripts/prepare_data_enhanced.py --input "data/raw/$filename"
                    ;;
                3)
                    echo ""
                    echo "Document Processing Options:"
                    echo "  1) Email files (.mbox format)"
                    echo "  2) PDF documents"
                    echo "  3) Word documents (.docx)"
                    echo "  4) PowerPoint presentations (.pptx)"
                    echo "  5) Excel spreadsheets (.xlsx)"
                    echo "  6) Plain text files (.txt)"
                    echo "  7) Process ALL file types"
                    echo ""
                    echo "Place your files in: data/to_process/"
                    ls -la data/to_process/ 2>/dev/null | head -10
                    echo ""
                    read -p "Select option [1-7]: " doc_choice
                    
                    case $doc_choice in
                        1) TYPE="email" ;;
                        2) TYPE="pdf" ;;
                        3) TYPE="docx" ;;
                        4) TYPE="pptx" ;;
                        5) TYPE="xlsx" ;;
                        6) TYPE="txt" ;;
                        7) TYPE="all" ;;
                        *) TYPE="all" ;;
                    esac
                    
                    echo "Processing $TYPE files..."
                    python3 scripts/prepare_universal_data.py --type $TYPE
                    ;;
                4)
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
