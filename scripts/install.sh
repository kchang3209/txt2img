#!/bin/bash
set -e

pip install -r requirements.txt

# Reinstall vLLM to ensure the correct binary is used.
pip install \
    --force-reinstall \
    --no-cache-dir \
    vllm==0.24.0 \
    --extra-index-url https://pytorch.org