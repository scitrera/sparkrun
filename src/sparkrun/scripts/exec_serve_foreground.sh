#!/bin/bash
set -uo pipefail

echo "Executing serve command in container {container_name}..."
docker exec {container_name} bash -c '{full_cmd}'
