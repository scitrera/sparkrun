# sparkrun

**One command to rule them all**

Launch, manage, and stop inference workloads on one or more NVIDIA DGX Spark systems — no Slurm, no Kubernetes, no fuss.

sparkrun is a unified CLI for running LLM inference on DGX Spark. Point it at your hosts, pick a recipe, and go. It
handles container orchestration, InfiniBand/RDMA detection, model distribution, and multi-node tensor parallelism across
your Spark cluster automatically.

sparkrun does not need to run on a member of the cluster. You can coordinate one or more DGX Sparks from any Linux
machine with SSH access.

**[Full documentation at sparkrun.dev](https://sparkrun.dev)** · **[Browse recipes on Spark Arena](https://spark-arena.com)**

## Install

```bash
# uv is preferred mechanism for managing python environments
# To install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

# automatic installation via uvx (manages virtual environment and
# creates alias in your shell, sets up autocomplete too!)
uvx sparkrun setup install
```

<details>
<summary>Alternative: manual pip install</summary>

```bash
pip install sparkrun
# or
uv pip install sparkrun
```

With a manual install you will need to run `sparkrun setup completion` separately for tab completion.

</details>

## Quick Start

```bash
# Save your hosts once
sparkrun cluster create mylab --hosts 192.168.11.13,192.168.11.14 -d "My DGX Spark lab"
sparkrun cluster set-default mylab

# Set up passwordless SSH mesh
sparkrun setup ssh

# Run an inference workload
sparkrun run qwen3-1.7b-vllm

# Multi-node tensor parallel (TP maps to node count on DGX Spark)
sparkrun run qwen3-1.7b-vllm --tp 2

# Override settings on the fly
sparkrun run qwen3-1.7b-vllm --port 9000 --gpu-mem 0.8
sparkrun run qwen3-1.7b-vllm --tp 2 -o max_model_len=8192

# GGUF quantized models via llama.cpp
sparkrun run qwen3-1.7b-llama-cpp
```

sparkrun launches jobs in the background (detached containers) and follows logs. **Ctrl+C detaches from
logs — it never kills your inference job.** Your model keeps serving.

```bash
# Re-attach to logs
sparkrun logs qwen3-1.7b-vllm

# Stop a workload
sparkrun stop qwen3-1.7b-vllm

# Check cluster status
sparkrun status
```

### Browse and inspect recipes

```bash
# List all available recipes
sparkrun list

# Search by name or model
sparkrun search qwen3

# Inspect a recipe with VRAM estimation
sparkrun show nemotron3-nano-30b-nvfp4-vllm
```

The VRAM estimator auto-detects model architecture from HuggingFace and tells you whether your configuration fits within
DGX Spark's 128 GB unified memory before you launch.

## Spark Arena

[Spark Arena](https://spark-arena.com) is the community recipe hub for DGX Spark. Browse tested recipes,
benchmark results, and one-click launch configs — then run them directly with sparkrun.

```bash
# Run a recipe from Spark Arena using its short link
sparkrun run @spark-arena/<recipe-id>

# Inspect a Spark Arena recipe with VRAM estimation
sparkrun show @spark-arena/<recipe-id>

# Save a Spark Arena recipe locally for customization
sparkrun show @spark-arena/<recipe-id> --save my-recipe.yaml
```

Spark Arena recipes are also available through sparkrun's default registries — `sparkrun list` includes
recipes from Spark Arena automatically. Visit [spark-arena.com](https://spark-arena.com) to browse
the full catalog, view benchmark results, and copy recipe links.

### Custom registries

sparkrun supports additional git-based registries for private or community recipes.

```bash
sparkrun registry list
sparkrun registry add https://github.com/myorg/spark-recipes.git
sparkrun registry update
```

See the [registries documentation](https://sparkrun.dev/recipes/registries/) for details.

## Supported Runtimes

| Runtime | Engine | Multi-Node |
|---------|--------|------------|
| **vllm** | [vLLM](https://github.com/vllm-project/vllm) | Distributed (default) or Ray |
| **sglang** | [SGLang](https://github.com/sgl-project/sglang) | Native distributed |
| **llama-cpp** | [llama.cpp](https://github.com/ggml-org/llama.cpp) | Solo (experimental RPC multi-node) |
| **eugr-vllm** | vLLM via [eugr](https://github.com/eugr/spark-vllm-docker) | Ray (with custom builds/mods) |

Each DGX Spark has one GPU with 128 GB unified memory, so tensor parallelism maps directly to node count:
`--tp 2` means 2 hosts. See the [runtime documentation](https://sparkrun.dev/cli/run/) for details.

## Prerequisites

- **SSH**: Passwordless SSH from your control machine to every cluster node. Run `sparkrun setup ssh` for easy setup.
- **Docker**: The SSH user must be in the `docker` group on every node (`sudo usermod -aG docker "$USER"`).

See the [getting started guide](https://sparkrun.dev/getting-started/quick-start/) for full setup instructions.

## Recipes

A recipe is a YAML file describing an inference workload — model, container image, runtime, and defaults:

```yaml
model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
runtime: vllm
container: scitrera/dgx-spark-vllm:0.16.0-t5

defaults:
  port: 8000
  tensor_parallel: 1
  gpu_memory_utilization: 0.8
  max_model_len: 200000
  served_model_name: nemotron3-30b-a3b
```

Any default can be overridden at launch time with `-o key=value` or dedicated flags like `--port`, `--tp`, `--gpu-mem`.
See [RECIPES.md](./RECIPES.md) for the full recipe format specification.

## CLI Reference

Run `sparkrun <command> --help` for full options on any command. Detailed documentation for each command
is available at [sparkrun.dev/cli](https://sparkrun.dev/cli/overview/).

### Workload commands

| Command | Description |
|---------|-------------|
| `sparkrun run <recipe>` | Launch an inference workload |
| `sparkrun stop <recipe>` | Stop a running workload |
| `sparkrun logs <recipe>` | Re-attach to workload logs |
| `sparkrun status` | Show running containers and cluster status |
| `sparkrun benchmark <recipe>` | Run → benchmark → stop (auto flow) |

Common options: `--hosts` / `-H`, `--cluster`, `--tp`, `--port`, `--gpu-mem`, `-o key=value`, `--dry-run`.

### Recipe commands

| Command | Description |
|---------|-------------|
| `sparkrun list [query]` | List available recipes |
| `sparkrun show <recipe>` | Show recipe details + VRAM estimate |
| `sparkrun search <query>` | Search recipes by name/model/description |
| `sparkrun recipe validate <recipe>` | Validate a recipe file |
| `sparkrun recipe vram <recipe>` | Estimate VRAM usage |

### Registry commands

| Command | Description |
|---------|-------------|
| `sparkrun registry list` | List configured registries |
| `sparkrun registry add <url>` | Add a registry from a repo manifest |
| `sparkrun registry remove <name>` | Remove a registry |
| `sparkrun registry enable/disable <name>` | Toggle a registry |
| `sparkrun registry update [name]` | Update registries from git |

### Cluster commands

| Command | Description |
|---------|-------------|
| `sparkrun cluster create <name>` | Create a named cluster |
| `sparkrun cluster list` | List all saved clusters |
| `sparkrun cluster show <name>` | Show cluster details |
| `sparkrun cluster set-default <name>` | Set the default cluster |
| `sparkrun cluster status` | Show running workloads on a cluster |

### Tune commands

| Command | Description |
|---------|-------------|
| `sparkrun tune sglang <recipe>` | Tune SGLang fused MoE Triton kernels |
| `sparkrun tune vllm <recipe>` | Tune vLLM fused MoE Triton kernels |

### Setup commands

| Command | Description |
|---------|-------------|
| `sparkrun setup install` | Install sparkrun as a uv tool + tab-completion + registry sync |
| `sparkrun setup update` | Update sparkrun + registries |
| `sparkrun setup ssh` | Set up passwordless SSH mesh across hosts |
| `sparkrun setup completion` | Install shell tab-completion |
| `sparkrun setup cx7` | Configure ConnectX-7 NICs |
| `sparkrun setup fix-permissions` | Fix root-owned HF cache files |
| `sparkrun setup clear-cache` | Drop Linux page cache on cluster hosts |

## About

sparkrun provides a unified tool for running inference on DGX Spark systems without Slurm or Kubernetes coordination. It
is intended to be donated to a future community organization.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
