#!/bin/bash
set -euo pipefail

CLONE_DIR="/tmp/sglang_src"

echo "Cloning SGLang benchmark scripts (sparse checkout)..."
rm -rf "$CLONE_DIR"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/sgl-project/sglang.git "$CLONE_DIR"
cd "$CLONE_DIR"
git sparse-checkout set benchmark/kernels/fused_moe_triton

echo "Benchmark scripts ready at $CLONE_DIR"
