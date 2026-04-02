#!/bin/bash

# Set parameters with defaults
MODEL=${1:-"Qwen/Qwen3-4B"}
PEFT_PATH=${2:-"adapter_checkpoint/"}
TEST_DATA_PATH=${3:-"data/public_test.json"}
OUTPUT_PATH=${4:-"output.json"}

# Run inference and save to output.json
echo "[INFO] Running inference..."
python3 inference.py --model_path $MODEL --adapter_checkpoint_path $PEFT_PATH --test_data_path $TEST_DATA_PATH --output_path $OUTPUT_PATH --batch_size 64