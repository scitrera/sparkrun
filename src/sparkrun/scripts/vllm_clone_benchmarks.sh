#!/bin/bash
set -euo pipefail

CLONE_DIR="/tmp/vllm_src"

echo "Cloning vLLM benchmark scripts (sparse checkout)..."
rm -rf "$CLONE_DIR"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/vllm-project/vllm.git "$CLONE_DIR"
cd "$CLONE_DIR"
git sparse-checkout set benchmarks/kernels

echo "Benchmark scripts ready at $CLONE_DIR"
