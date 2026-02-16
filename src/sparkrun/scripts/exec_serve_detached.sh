#!/bin/bash
set -uo pipefail

echo "Executing serve command in container {container_name} (detached)..."
docker exec {container_name} bash -c "nohup bash -c '{full_cmd}' > /tmp/sparkrun_serve.log 2>&1 &"

# Wait for process to start and produce initial output
sleep 3

echo "============================================================"
echo "Initial log output:"
docker exec {container_name} tail -n 30 /tmp/sparkrun_serve.log 2>/dev/null || echo "(no log output yet)"
echo "============================================================"
echo "Serve command launched in background."
echo "To follow logs:  ssh <host> docker exec {container_name} tail -f /tmp/sparkrun_serve.log"