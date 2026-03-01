---
name: setup
description: Install sparkrun and configure DGX Spark clusters
---

<Purpose>
Provides complete reference for installing sparkrun, creating and managing cluster configurations, setting up SSH mesh for multi-node inference, configuring CX7 networking, fixing file permissions, clearing page cache, and configuring the sparkrun environment on NVIDIA DGX Spark systems.
</Purpose>

<Use_When>
- User wants to install or update sparkrun
- User wants to create, modify, or manage cluster configurations
- User wants to set up SSH for multi-node inference
- User wants to configure CX7 network interfaces
- User wants to fix file permissions on cluster hosts
- User wants to clear page cache on cluster hosts
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

# Update to latest version (and registries)
sparkrun setup update

# Update sparkrun only (skip registry sync)
sparkrun setup update --no-update-registries
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

## CX7 Networking

Configure ConnectX-7 network interfaces on cluster hosts for high-speed transfers.

```bash
# Auto-detect CX7 interfaces and configure with defaults
sparkrun setup cx7 --cluster <name>
sparkrun setup cx7 --hosts <ip1>,<ip2>

# Override subnets
sparkrun setup cx7 --cluster <name> --subnet1 192.168.11.0/24 --subnet2 192.168.12.0/24

# Force reconfiguration and set MTU
sparkrun setup cx7 --cluster <name> --force --mtu 9000

# Dry-run
sparkrun setup cx7 --cluster <name> --dry-run
```

Requires passwordless sudo on target hosts. Will prompt for sudo password if needed.

## Fix File Permissions

Fix file ownership in HuggingFace cache directories on cluster hosts. Docker containers create files as root, leaving the normal user unable to manage the cache.

```bash
# Fix permissions on default cache directory
sparkrun setup fix-permissions --cluster <name>
sparkrun setup fix-permissions --hosts <ip1>,<ip2>

# Custom cache directory
sparkrun setup fix-permissions --cluster <name> --cache-dir /data/hf-cache

# Install sudoers entry for passwordless future runs
sparkrun setup fix-permissions --cluster <name> --save-sudo

# Dry-run
sparkrun setup fix-permissions --cluster <name> --dry-run
```

## Clear Page Cache

Drop the Linux page cache on cluster hosts to free memory for inference.

```bash
# Clear cache on cluster hosts
sparkrun setup clear-cache --cluster <name>
sparkrun setup clear-cache --hosts <ip1>,<ip2>

# Install sudoers entry for passwordless future runs
sparkrun setup clear-cache --cluster <name> --save-sudo

# Dry-run
sparkrun setup clear-cache --cluster <name> --dry-run
```

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
- `sparkrun setup cx7` requires passwordless sudo; use `--force` to reconfigure already-valid hosts
- `sparkrun setup fix-permissions` and `clear-cache` try non-interactive sudo first, then prompt if needed
- Use `--save-sudo` to install scoped sudoers entries for passwordless future runs
</Important_Notes>

Task: {{ARGUMENTS}}
