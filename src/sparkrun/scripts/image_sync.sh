#!/bin/bash
set -uo pipefail
if docker image inspect "{image}" >/dev/null 2>&1; then
    echo "Image already available: {image}"
    exit 0
fi
echo "Pulling image: {image}..."
docker pull "{image}"
