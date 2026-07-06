#!/bin/bash

set -e

MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"
LOCAL_DIR="./weights/qwen3-coder-30b"

echo "Downloading $MODEL_NAME..."
hf download "$MODEL_NAME" \
    --local-dir "$LOCAL_DIR" \
    --quiet

echo "Done!"