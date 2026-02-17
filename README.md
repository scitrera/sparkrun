# sparkrun

**One command to rule them all**

Launch, manage, and stop inference workloads on one or more NVIDIA DGX Spark systems — no Slurm, no Kubernetes, no fuss.

sparkrun is a unified CLI for running LLM inference on DGX Spark. Point it at your hosts, pick a recipe, and go. It
handles container orchestration, InfiniBand/RDMA detection, model distribution, and multi-node tensor parallelism across
your Spark cluster automatically.

sparkrun does not need to run on a member of the cluster. You can coordinate one or more DGX Sparks from any Linux
machine with SSH access.

```
# uv is preferred mechanism for managing python environments
pip install uv

# automatic installation via uvx (installs virtual environment and 
# creates alias in your shell, sets up autocomplete too!)
uvx sparkrun setup install

# power users may prefer to manage their own setup -- so there is always:
pip install sparkrun
-- or --
uv pip install sparkrun
```

## Quick Start

### Run an inference job

```bash
# Single node
sparkrun run nemotron3-nano-30b-nvfp4-vllm --solo

# Multi-node (2-node tensor parallel)
sparkrun run nemotron3-nano-30b-nvfp4-vllm --hosts 192.168.11.13,192.168.11.14

# Override settings on the fly
sparkrun run nemotron3-nano-30b-nvfp4-vllm --hosts 192.168.11.13 --port 9000 --gpu-mem 0.9
sparkrun run nemotron3-nano-30b-nvfp4-vllm -H 192.168.11.13,192.168.11.14 -o max_model_len=8192
```

sparkrun always launches jobs in the background (detached containers) and then follows logs. **Ctrl+C detaches from
logs — it never kills your inference job.** Your model keeps serving.

### Save a cluster config

```bash
# Save your hosts once
sparkrun cluster create mylab --hosts 192.168.11.13,192.168.11.14 -d "My DGX Spark lab"
sparkrun cluster set-default mylab

# Now just run — hosts are automatic
sparkrun run nemotron3-nano-30b-nvfp4-vllm
```

### Tab completion

```bash
sparkrun setup completion          # auto-detects your shell
sparkrun setup completion --shell zsh
```

After restarting your shell, recipe names, cluster names, and subcommands all tab-complete.

### Inspect a recipe

```bash
sparkrun show nemotron3-nano-30b-nvfp4-vllm
```

```
Name:         nemotron3-nano-30b-nvfp4
Description:  NVIDIA Nemotron 3 Nano 30B (upstream NVFP4) -- cluster or solo
Runtime:      vllm
Model:        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
Container:    scitrera/dgx-spark-vllm:0.16.0-t5
Nodes:        1 - unlimited
Repository:   Local

Defaults:
  gpu_memory_utilization: 0.8
  max_model_len: 200000
  port: 8000
  served_model_name: nemotron3-30b-a3b
  tensor_parallel: 1

VRAM Estimation:
  Model dtype:      nvfp4
  Model params:     30,000,000,000
  KV cache dtype:   bfloat16
  Architecture:     52 layers, 2 KV heads, 128 head_dim
  Model weights:    19.56 GB
  KV cache:         9.92 GB (max_model_len=200,000)
  Tensor parallel:  1
  Per-GPU total:    29.48 GB
  DGX Spark fit:    YES

  GPU Memory Budget:
    gpu_memory_utilization: 80%
    Usable GPU memory:     96.8 GB (121 GB x 80%)
    Available for KV:      77.2 GB
    Max context tokens:    1,557,583
    Context multiplier:    7.8x (vs max_model_len=200,000)
```

The VRAM estimator auto-detects model architecture from HuggingFace and tells you whether your configuration fits within
DGX Spark's 128 GB unified memory before you launch.

### Custom recipe registries

```bash
# See what's configured
sparkrun recipe registries

# Add a community or private registry
sparkrun recipe add-registry myteam \
  --url https://github.com/myorg/spark-recipes.git \
  --subpath recipes

# Update all registries
sparkrun recipe update

# Search across all registries
sparkrun search qwen3
```

### Manage running workloads

```bash
# Re-attach to logs (Ctrl+C is always safe) -- NOTE: finds cluster by combination of hosts, model, and runtime
sparkrun log nemotron3-nano-30b-nvfp4-vllm --cluster mylab

# Stop a workload -- NOTE: finds cluster by combination of hosts, model, and runtime
sparkrun stop nemotron3-nano-30b-nvfp4-vllm --cluster mylab
```

## Supported Runtimes

### vLLM

First-class support for [vLLM](https://github.com/vllm-project/vllm). Solo and multi-node clustering via Ray. Works with
ready-built images (e.g. `scitrera/dgx-spark-vllm`). Also works with other images including those build from eugr's repo
and/or NVIDIA images.

### SGLang

First-class support for [SGLang](https://github.com/sgl-project/sglang). Solo and multi-node clustering via SGLang's
native distributed backend (`--dist-init-addr`, `--nnodes`, `--node-rank`). Works with ready-built images (e.g.
`scitrera/dgx-spark-sglang`). Should also work with other sglang images, but there seem to be a lot fewer sglang images
around than vllm images.

### eugr-vllm (compatibility runtime)

Full compatibility with [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker). This runtime delegates
entirely to eugr's scripts — mods, local builds, and all eugr-specific features work natively because sparkrun calls
their code directly rather than reimplementing it.

Use this when you need a nightly vLLM build, custom modifications, or anything that requires building containers locally
from eugr's repo.

The recipe format for sparkrun is designed to be mostly compatible with eugr's (more like a v2 format) -- sparkrun will
translate any variations in recipe format to the eugr repo format automatically. Changes were mostly to ensure greater
compatibility with multiple runtimes and to reduce redundancy (somewhat). The full command listing is preserved to
ensure greater compatibility, but long-term, runtime implementations should be able to generate commands.

```yaml
# eugr-vllm recipe example
runtime: eugr-vllm
model: my-org/custom-model
container: vllm-node-tf5
runtime_config:
  mods: [ my-custom-mod ]
  build_args: [ --some-flag ]
```

## How It Works

**Recipes** are YAML files that describe an inference workload: the model, container image, runtime, and default
parameters. sparkrun ships bundled recipes and supports custom registries (any git repo with YAML files). Sparkrun includes
limited recipes and otherwise also includes the eugr repo as a default registry (which also delegates running to eugr's repo also...). 
The idea in the long-run is to merge recipes from multiple registries into a single unified catalog. And be able to run them
even if they were designed for different runtimes (e.g. vLLM vs SGLang) without needing to worry about the underlying command differences.

**Runtimes** are plugins that know how to launch a specific inference engine. sparkrun discovers them via Python entry
points, so custom runtimes can be added by installing a package.

**Orchestration** is handled over SSH. sparkrun detects InfiniBand/RDMA interfaces on your hosts, distributes container
images and models from local to remote (using the ethernet interfaces of the RDMA interfaces for fast transfers when available), configures NCCL environment
variables, and launches containers with the right networking.

Each DGX Spark has one GPU, so tensor parallelism maps directly to node count: `--tp 2` means 2 hosts.

## Recipes

A recipe is a YAML file:

```yaml
model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
runtime: vllm
min_nodes: 2
container: scitrera/dgx-spark-vllm:0.16.0-t5

metadata:
  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4)
  maintainer: scitrera.ai <open-source-team@scitrera.com>

defaults:
  port: 8000
  tensor_parallel: 1
  gpu_memory_utilization: 0.8
  max_model_len: 200000
  served_model_name: nemotron3-30b-a3b

command: |
  vllm serve {model} \
      --served-model-name {served_model_name} \
      --max-model-len {max_model_len} \
      --gpu-memory-utilization {gpu_memory_utilization} \
      -tp {tensor_parallel} \
      --host {host} --port {port}
```

Any default can be overridden at launch time with `-o key=value` or dedicated flags like `--port`, `--tp`, `--gpu-mem`.

## CLI Reference

| Command                           | Description                              |
|-----------------------------------|------------------------------------------|
| `sparkrun run <recipe>`           | Launch an inference workload             |
| `sparkrun stop <recipe>`          | Stop a running workload                  |
| `sparkrun log <recipe>`           | Re-attach to workload logs               |
| `sparkrun list`                   | List available recipes                   |
| `sparkrun show <recipe>`          | Show recipe details + VRAM estimate      |
| `sparkrun search <query>`         | Search recipes by name/model/description |
| `sparkrun cluster create`         | Save a named cluster definition          |
| `sparkrun cluster list`           | List saved clusters                      |
| `sparkrun cluster set-default`    | Set the default cluster                  |
| `sparkrun recipe registries`      | List configured recipe registries        |
| `sparkrun recipe add-registry`    | Add a custom recipe registry             |
| `sparkrun recipe remove-registry` | Remove a recipe registry                 |
| `sparkrun recipe update`          | Update registries from git               |
| `sparkrun recipe validate`        | Validate a recipe file                   |
| `sparkrun recipe vram`            | Estimate VRAM usage for a recipe         |
| `sparkrun setup completion`       | Install shell tab-completion             |

## Roadmap

- `sparkrun setup` subcommands for basic system configuration, ConnectX-7 NIC setup, and SSH mesh provisioning
- Additional bundled recipes for popular models
- Health checks and status monitoring for running workloads

## About

sparkrun provides a unified tool for running inference on DGX Spark systems without Slurm or Kubernetes coordination. It
is intended to be donated to a future community organization.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
