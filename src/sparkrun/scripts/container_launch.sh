#!/bin/bash
set -uo pipefail

echo "Cleaning up existing container: {container_name}"
{cleanup_cmd}

echo "Launching container: {container_name}"
echo "Image: {image}"
{run_cmd}

echo "Container {container_name} launched successfully"
