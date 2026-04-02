#!/bin/bash

# Set parameters with defaults
MODEL=${1:-"Qwen/Qwen3-4B"}
PEFT_PATH=${2:-"adapter_checkpoint/"}
TEST_DATA_PATH=${3:-"data/public_test.json"}
OUTPUT_PATH=${4:-"output.json"}

# Run inference and save to output.json
# (tested with ~6GB VRAM usage on batch_size=16 for Qwen/Qwen3-4B)
echo "[INFO] Running inference..."
python3 inference.py --model_path $MODEL --adapter_checkpoint_path $PEFT_PATH --test_dataset $TEST_DATA_PATH --output_path $OUTPUT_PATH --batch_size 16
echo "[INFO] Done! Output saved to $OUTPUT_PATH"