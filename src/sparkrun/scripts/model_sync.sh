#!/bin/bash
set -uo pipefail
echo "Checking model cache for {model_id}..."
SAFE_NAME=$(echo "{model_id}" | tr '/' '--')
CACHE_PATH="{cache}/hub/models--$SAFE_NAME"

if [ -d "$CACHE_PATH" ] && [ "$(ls -A "$CACHE_PATH" 2>/dev/null)" ]; then
    echo "Model already cached: {model_id}"
    exit 0
fi

echo "Downloading model: {model_id}..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "{model_id}" --cache-dir "{cache}"
else
    echo "ERROR: huggingface-cli not available on this host" >&2
    exit 1
fi
