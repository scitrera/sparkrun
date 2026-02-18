---
name: setup
description: Install sparkrun and configure DGX Spark clusters
---

<Purpose>
Provides complete reference for installing sparkrun, creating and managing cluster configurations, setting up SSH mesh for multi-node inference, and configuring the sparkrun environment on NVIDIA DGX Spark systems.
</Purpose>

<Use_When>
- User wants to install or update sparkrun
- User wants to create, modify, or manage cluster configurations
- User wants to set up SSH for multi-node inference
- User asks about sparkrun configuration or setup
- User is getting started with DGX Spark inference for the first time
</Use_When>

<Do_Not_Use_When>
- User wants to run, stop, or monitor workloads -- use the run skill instead
- User wants to manage recipe registries or create recipes -- use the registry skill instead
</Do_Not_Use_When>

<Steps>

## Installation

```bash
# Ensure that uv is installed
uv --version

# uv can be installed with (IF NEEDED)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install sparkrun as a CLI tool
uvx sparkrun setup install

# Update to latest version (not needed after initial install)
sparkrun setup update
```

## Cluster Management

Clusters are named host groups saved in `~/.config/sparkrun/clusters/`.

```bash
# Create a cluster (first host = head node)
sparkrun cluster create <name> --hosts <ip1>,<ip2>,... [-d "description"] [--user <ssh_user>]

# Set as default (used when --hosts/--cluster not specified)
sparkrun cluster set-default <name>

# View clusters
sparkrun cluster list
sparkrun cluster show <name>
sparkrun cluster default

# Modify
sparkrun cluster update <name> --hosts <new_hosts> [--user <user>] [-d "desc"]
sparkrun cluster delete <name>
sparkrun cluster unset-default
```

## SSH Setup

Multi-node inference requires passwordless SSH between all hosts. sparkrun bundles a mesh setup script.

```bash
# Set up SSH mesh across cluster hosts (interactive -- prompts for passwords)
sparkrun setup ssh --cluster <name>
sparkrun setup ssh --hosts <ip1>,<ip2> [--user <username>]

# Include extra hosts (e.g. control machine) in the mesh
sparkrun setup ssh --cluster <name> --extra-hosts <control_ip>

# Exclude the local machine from the mesh
sparkrun setup ssh --cluster <name> --no-include-self

# Dry-run to see what would happen
sparkrun setup ssh --cluster <name> --dry-run
```

**IMPORTANT:** The SSH setup script runs interactively (prompts for passwords on first connection). Do NOT capture its output -- let it pass through to the terminal.

## Configuration

Config file: `~/.config/sparkrun/config.yaml`

Key settings:
- `cluster.hosts`: Default host list (used when no --hosts/--cluster given)
- `ssh.user`: Default SSH username
- `ssh.key`: Path to SSH private key
- `ssh.options`: Additional SSH options list
- `cache_dir`: sparkrun cache directory (default: `~/.cache/sparkrun`)
- `hf_cache_dir`: HuggingFace cache directory (default: `~/.cache/huggingface`)

</Steps>

<Tool_Usage>
All sparkrun commands are executed via the Bash tool. No MCP tools are required.

When running SSH setup, the command is interactive and must be run with inherited stdio -- do NOT use `capture_output` or pipe through other commands.
</Tool_Usage>

<Important_Notes>
- Always create a cluster and set it as default for the user's lab setup
- The first host in a cluster is the **head node** for multi-node jobs
- SSH mesh must be set up before multi-node inference will work
- `sparkrun setup ssh` is interactive -- let it pass through to the terminal
- DGX Spark systems have 1 GPU per host, so `tensor_parallel` maps to node count
- `uv` is the recommended Python package manager; install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
</Important_Notes>

Task: {{ARGUMENTS}}
