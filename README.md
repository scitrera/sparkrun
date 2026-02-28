# sparkrun

**One command to rule them all**

Launch, manage, and stop inference workloads on one or more NVIDIA DGX Spark systems — no Slurm, no Kubernetes, no fuss.

sparkrun is a unified CLI for running LLM inference on DGX Spark. Point it at your hosts, pick a recipe, and go. It
handles container orchestration, InfiniBand/RDMA detection, model distribution, and multi-node tensor parallelism across
your Spark cluster automatically.

sparkrun does not need to run on a member of the cluster. You can coordinate one or more DGX Sparks from any Linux
machine with SSH access.

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

### Tab completion

> **Note:** If you installed via `sparkrun setup install`, tab completion is already set up — you can skip this step.

```bash
sparkrun setup completion          # auto-detects your shell
sparkrun setup completion --shell zsh
```

After restarting your shell, recipe names, cluster names, and subcommands all tab-complete.

### Save a cluster config

```bash
# Save your hosts once
sparkrun cluster create mylab --hosts 192.168.11.13,192.168.11.14 -d "My DGX Spark lab"
sparkrun cluster set-default mylab

# Now just run — hosts are automatic
sparkrun run nemotron3-nano-30b-nvfp4-vllm
```

### Run an inference job

```bash
# Single node vLLM (Note that minimum nodes / parallelism is configured by the recipe)
sparkrun run qwen3-1.7b-vllm

# Multi-node (2-node tensor parallel) -- using your default two node cluster
sparkrun run qwen3-1.7b-vllm --tp 2

# Override settings on the fly
sparkrun run qwen3-1.7b-vllm --hosts 192.168.11.14 --port 9000 --gpu-mem 0.8
sparkrun run qwen3-1.7b-vllm --tp 2 -H 192.168.11.13,192.168.11.14 -o max_model_len=8192

# GGUF quantized models via llama.cpp
sparkrun run qwen3-1.7b-llama-cpp
```

sparkrun always launches jobs in the background (detached containers) and then follows logs. **Ctrl+C detaches from
logs — it never kills your inference job.** Your model keeps serving.

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
sparkrun registry list

# Add a community or private registry
sparkrun registry add https://github.com/myorg/spark-recipes.git

# Update all registries
sparkrun registry update

# Search across all registries
sparkrun search qwen3
```

### Manage running workloads

```bash
# Re-attach to logs (Ctrl+C is always safe) -- NOTE: finds cluster by combination of hosts, model, and runtime
sparkrun logs nemotron3-nano-30b-nvfp4-vllm --cluster mylab

# Stop a workload -- NOTE: finds cluster by combination of hosts, model, and runtime
sparkrun stop nemotron3-nano-30b-nvfp4-vllm --cluster mylab

# If you launched with --tp (modifying the recipe default), e.g.:
sparkrun run nemotron3-nano-30b-nvfp4-vllm --tp 2
# then pass --tp so stop/logs resolve the same cluster ID as run:
sparkrun stop nemotron3-nano-30b-nvfp4-vllm --tp 2
sparkrun logs nemotron3-nano-30b-nvfp4-vllm --tp 2
# TIP: you can just press up and modify "run" to "stop"
```

## Supported Runtimes

### vLLM

First-class support for [vLLM](https://github.com/vllm-project/vllm) with two multi-node variants:

- **`vllm-distributed`** (default) — uses vLLM's built-in distributed backend. Each node runs the full serve command with
  node-rank arguments.
- **`vllm-ray`** — uses Ray head/worker orchestration. A Ray cluster is formed first, then the serve command runs on the
  head node.

Write `runtime: vllm` in a recipe and sparkrun resolves it automatically: `vllm-distributed` by default, or `vllm-ray`
when Ray hints are present (e.g. `distributed_executor_backend: ray` in defaults). Works with ready-built images
(e.g. `scitrera/dgx-spark-vllm`). Also works with other images including those built from eugr's repo and/or NVIDIA images.

### SGLang

First-class support for [SGLang](https://github.com/sgl-project/sglang). Solo and multi-node clustering via SGLang's
native distributed backend (`--dist-init-addr`, `--nnodes`, `--node-rank`). Works with ready-built images (e.g.
`scitrera/dgx-spark-sglang`). Should also work with other sglang images, but there seem to be a lot fewer sglang images
around than vllm images.

### llama.cpp

Support for [llama.cpp](https://github.com/ggml-org/llama.cpp) via `llama-server`. Solo mode with GGUF quantized models.
Loads models directly from HuggingFace (e.g. `Qwen/Qwen3-1.7B-GGUF:Q4_K_M`). Lightweight alternative to vLLM/SGLang
for smaller models or constrained environments.

GGUF models use colon syntax to select a quantization variant: `model: Qwen/Qwen3-1.7B-GGUF:Q8_0`. sparkrun
pre-downloads only the matching quant files and resolves the local cache path so the container doesn't need to
re-download at serve time.

**Experimental**: Multi-node inference via llama.cpp's RPC backend. Worker nodes run `rpc-server` and
the head node connects via `--rpc`. This is still evolving both upstream and in sparkrun and should be considered 
experimental. Note that the fastest DGX Spark interconnect communication will be via NCCL and RoCE -- and the 
llama.cpp RPC mechanism involves a lot more overhead. 

### eugr-vllm

Extends `vllm-ray` with [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) container
builds and mod support. All sparkrun orchestration features work natively — multi-node Ray clustering, container
distribution, InfiniBand detection, log following, and cluster stop — while eugr-specific features (local container
builds via `build-and-copy.sh`, mod application) are integrated into the launch pipeline.

Use this when you need a nightly vLLM build, custom modifications, or anything that requires building containers locally
from eugr's repo. For prebuilt images without mods or custom builds, use the standard `vllm` runtime instead.

The recipe format for sparkrun is designed to be mostly compatible with eugr's format — sparkrun translates any
variations automatically. Recipes with `recipe_version: "1"` or eugr-specific fields (`build_args`, `mods`) are
auto-detected as `eugr-vllm`.

```yaml
# eugr-vllm recipe example
runtime: eugr-vllm
model: my-org/custom-model
container: vllm-node
runtime_config:
  mods: [ my-custom-mod ]
  build_args: [ --some-flag ]
```

## How It Works

**Recipes** are YAML files that describe an inference workload: the model, container image, runtime, and default
parameters. sparkrun ships bundled recipes and supports custom registries (any git repo with YAML files). Sparkrun
includes limited recipes and also includes the eugr repo as a default registry. Recipes from eugr's format are
auto-detected and run natively through sparkrun's orchestration pipeline. The idea in the long run is to merge recipes
from multiple registries into a single unified catalog and be able to run them even if they were designed for different
runtimes (e.g. vLLM vs SGLang) without needing to worry about the underlying command differences. See the
[RECIPES](./RECIPES.md) specification file for more details.

**Runtimes** are plugins that know how to launch a specific inference engine. sparkrun discovers them via Python entry
points, so custom runtimes can be added by installing a package. When a recipe omits the `runtime` field, sparkrun
auto-detects it from the `command` prefix — `vllm serve` → vLLM, `sglang serve` → SGLang, `llama-server` → llama.cpp.
See [RECIPES.md](./RECIPES.md#automatic-runtime-detection) for the full detection rules.

**Orchestration** is handled over SSH. sparkrun detects InfiniBand/RDMA interfaces on your hosts, distributes container
images and models from local to remote (using the ethernet interfaces of the RDMA interfaces for fast transfers when
available), configures NCCL environment variables, and launches containers with the right networking.

Each DGX Spark has one GPU, so tensor parallelism maps directly to node count: `--tp 2` means 2 hosts.

### SSH Prerequisites

All multi-node orchestration relies on SSH. At minimum, you need **passwordless SSH from your control machine
to every cluster node**. sparkrun pulls container images and models locally and pushes them to each node
directly, so node-to-node SSH is not strictly required for the default workflow.

That said, setting up a **full SSH mesh** (every host can reach every other host) is recommended — it enables
alternative distribution strategies and is generally useful for cluster administration.

The easiest way to set this up is `sparkrun setup ssh`, which creates a full mesh across your cluster
hosts **and** the control machine (included automatically via `--include-self`, on by default):

```bash
# Set up passwordless SSH mesh across your cluster + this machine
sparkrun setup ssh --hosts 192.168.11.13,192.168.11.14 --user ubuntu

# Or use a saved cluster
sparkrun setup ssh --cluster mylab

# Or if you've set your default cluster -- it'll just use that
sparkrun setup ssh

# Add extra hosts beyond the cluster (e.g. a jump host)
sparkrun setup ssh --cluster mylab --extra-hosts 10.0.0.99

# Exclude the control machine from the mesh
sparkrun setup ssh --cluster mylab --no-include-self
```

You will be prompted for passwords on first connection to each host. After that, every host in the
mesh can SSH to every other host without passwords.

<details>
<summary>Manual SSH setup (without sparkrun setup ssh)</summary>

If you prefer to set up SSH yourself, you need key-based auth from your control machine to each node:

```bash
# Generate a key if you don't have one
ssh-keygen -t ed25519

# Copy to each node
ssh-copy-id 192.168.11.13
ssh-copy-id 192.168.11.14
```

</details>

**SSH user**: By default sparkrun uses your current OS user for SSH. You can set a per-cluster user
with `sparkrun cluster create --user dgxuser` or `sparkrun cluster update --user dgxuser`, or override
per-command with `--user`.

<details>
<summary>For more advanced SSH configuration (non-default ports, identity files), use `~/.ssh/config`.</summary>

```
Host spark1
    HostName 192.168.11.13
    User dgxuser

Host spark2
    HostName 192.168.11.14
    User dgxuser
```

</details>

Solo mode (`--solo`) runs on a single host and still uses SSH unless the target is `localhost`.

### Docker Group

sparkrun launches containers via `docker` on each host. The SSH user must be a member of the `docker` group
on every cluster node:

```bash
sudo usermod -aG docker "$USER"
```

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

Recipes can also include an `env` block for environment variables injected into the container. Shell variable
references like `${HF_TOKEN}` are expanded from the control machine's environment, so you can forward secrets
without hardcoding them. See [RECIPES.md](./RECIPES.md) for the full recipe format specification.

### GGUF recipes (llama.cpp)

GGUF recipes use the `llama-cpp` runtime and specify a quantization variant with colon syntax:

```yaml
model: Qwen/Qwen3-1.7B-GGUF:Q8_0
runtime: llama-cpp
min_nodes: 1
max_nodes: 1
container: scitrera/dgx-spark-llama-cpp:latest

defaults:
  port: 8000
  host: 0.0.0.0
  n_gpu_layers: 99
  ctx_size: 8192

command: |
  llama-server \
      -hf {model} \
      --host {host} --port {port} \
      --n-gpu-layers {n_gpu_layers} \
      --ctx-size {ctx_size} \
      --flash-attn on --jinja --no-webui
```

When model pre-sync is enabled (the default), sparkrun downloads only the matching quant files locally, distributes
them to target hosts, and rewrites `-hf` to `-m` with the resolved container cache path so the container serves from
the local copy without re-downloading.

## CLI Reference

### Global options

| Option             | Description                 |
|--------------------|-----------------------------|
| `-v` / `--verbose` | Enable verbose/debug output |
| `--version`        | Show version and exit       |
| `--help`           | Show help for any command   |

### Workload commands

| Command                  | Description                  |
|--------------------------|------------------------------|
| `sparkrun run <recipe>`  | Launch an inference workload |
| `sparkrun stop <recipe>` | Stop a running workload      |
| `sparkrun logs <recipe>` | Re-attach to workload logs   |
| `sparkrun benchmark <recipe>` | Benchmark an inference recipe (auto: run → benchmark → stop) |

**`sparkrun run` options:**

| Option                       | Description                                              |
|------------------------------|----------------------------------------------------------|
| `--hosts` / `-H`             | Comma-separated host list (first = head)                 |
| `--hosts-file`               | File with hosts (one per line, `#` comments)             |
| `--cluster`                  | Use a saved cluster by name                              |
| `--solo`                     | Force single-node mode                                   |
| `--port`                     | Override serve port                                      |
| `--tp` / `--tensor-parallel` | Override tensor parallelism                              |
| `--gpu-mem`                  | Override GPU memory utilization (0.0-1.0)                |
| `--image`                    | Override container image  (not recommended)              |
| `--cache-dir`                | HuggingFace cache directory                              |
| `--option` / `-o`            | Override any recipe default: `-o key=value` (repeatable) |
| `--dry-run` / `-n`           | Show what would be done without executing                |
| `--foreground`               | Run in foreground (don't detach)                         |
| `--no-follow`                | Don't follow container logs after launch                 |
| `--no-sync-tuning`           | Skip syncing tuning configs from registries              |
| `--ray-port`                 | Ray GCS port (default: 46379) (`vllm-ray`/`eugr-vllm`)  |
| `--init-port`                | SGLang distributed init port (default: 25000)            |
| `--dashboard`                | Enable Ray dashboard on head node (`vllm-ray`/`eugr-vllm`) |
| `--dashboard-port`           | Ray dashboard port (default: 8265)                       |

**`sparkrun stop` options:**

| Option                       | Description                  |
|------------------------------|------------------------------|
| `--hosts` / `-H`             | Comma-separated host list    |
| `--hosts-file`               | File with hosts              |
| `--cluster`                  | Use a saved cluster by name  |
| `--all` / `-a`               | Stop all sparkrun containers (no recipe needed)          |
| `--tp` / `--tensor-parallel` | Match host trimming from run |
| `--dry-run` / `-n`           | Show what would be done      |

**`sparkrun logs` options:**

| Option                       | Description                                         |
|------------------------------|-----------------------------------------------------|
| `--hosts` / `-H`             | Comma-separated host list                           |
| `--hosts-file`               | File with hosts                                     |
| `--cluster`                  | Use a saved cluster by name                         |
| `--tp` / `--tensor-parallel` | Match host trimming from run                        |
| `--tail`                     | Number of existing log lines to show (default: 100) |

**`sparkrun benchmark` options:**

| Option                       | Description                                              |
|------------------------------|----------------------------------------------------------|
| `--hosts` / `-H`             | Comma-separated host list (first = head)                 |
| `--hosts-file`               | File with hosts (one per line, `#` comments)             |
| `--cluster`                  | Use a saved cluster by name                              |
| `--solo`                     | Force single-node mode                                   |
| `--tp` / `--tensor-parallel` | Override tensor parallelism                              |
| `--port`                     | Override serve port                                      |
| `--image`                    | Override container image                                 |
| `--cache-dir`                | HuggingFace cache directory                              |
| `--profile`                  | Benchmark profile name or file (`@registry/name` syntax) |
| `--framework`                | Override benchmarking framework (default: `llama-benchy`) |
| `--out` / `--output`         | Output file for results YAML                             |
| `--option` / `-o`            | Override benchmark args: `-o key=value` (repeatable)     |
| `--no-stop`                  | Don't stop inference after benchmarking                  |
| `--skip-run`                 | Skip launching inference (benchmark existing instance)   |
| `--sync-tuning`              | Sync tuning configs from registries before benchmarking  |
| `--timeout`                  | Benchmark timeout in seconds (default: 14400)            |
| `--dry-run` / `-n`           | Show what would be done without executing                |

### Tune commands

| Command                            | Description                                       |
|------------------------------------|---------------------------------------------------|
| `sparkrun tune sglang <recipe>`    | Tune SGLang fused MoE Triton kernels              |
| `sparkrun tune vllm <recipe>`      | Tune vLLM fused MoE Triton kernels                |

Run Triton kernel autotuning for MoE models. Generates optimal tile configs per TP size and saves
them for automatic use in future inference runs.

> **Note:** Tuning is computationally intensive — each TP size can take hours depending on model
> size. Use `--dry-run` to preview and `-j4` to parallelize across TP sizes.

```bash
# Tune SGLang kernels on localhost
sparkrun tune sglang qwen3.5-35b-bf16-sglang -H 127.0.0.1

# Tune vLLM kernels with 4 parallel jobs
sparkrun tune vllm qwen3-moe-vllm -H 127.0.0.1 -j4

# Tune specific TP sizes only
sparkrun tune sglang qwen3.5-35b-bf16-sglang -H 127.0.0.1 --tp 1 --tp 2
```

**Options:**

| Option                       | Description                                              |
|------------------------------|----------------------------------------------------------|
| `--hosts` / `-H`             | Target host (only the first host is used)                |
| `--cluster`                  | Use a saved cluster by name                              |
| `--tp`                       | TP size(s) to tune (repeatable; default: 1,2,4,8)        |
| `--parallel` / `-j`          | Run N tuning jobs concurrently (default: 1 = sequential) |
| `--image`                    | Override container image                                 |
| `--output-dir`               | Output directory for tuning configs                      |
| `--cache-dir`                | HuggingFace cache directory                              |
| `--skip-clone`               | Skip cloning benchmark scripts (if already in image)     |
| `--dry-run` / `-n`           | Show what would be done without executing                |

### Recipe commands

| Command                             | Description                                       |
|-------------------------------------|---------------------------------------------------|
| `sparkrun list [query]`             | List available recipes (alias)                    |
| `sparkrun show <recipe>`            | Show recipe details + VRAM estimate (alias)       |
| `sparkrun search <query>`           | Search recipes by name/model/description (alias)  |
| `sparkrun recipe list [query]`      | List available recipes from all registries        |
| `sparkrun recipe show <recipe>`     | Show detailed recipe information                  |
| `sparkrun recipe search <query>`    | Search for recipes by name, model, or description |
| `sparkrun recipe validate <recipe>` | Validate a recipe file                            |
| `sparkrun recipe vram <recipe>`     | Estimate VRAM usage for a recipe                  |

The `list`, `search`, `show` commands (both top-level aliases and `recipe` subcommands) accept `--all` / `-a`
to include recipes from hidden registries, and `--runtime` to filter by runtime type. `show` also accepts
`--save <path>` to save a copy of the recipe YAML and `--no-vram` to skip VRAM estimation.

**`sparkrun recipe vram` options:**

| Option                       | Description                               |
|------------------------------|-------------------------------------------|
| `--tp` / `--tensor-parallel` | Override tensor parallelism               |
| `--max-model-len`            | Override max sequence length              |
| `--gpu-mem`                  | Override gpu_memory_utilization (0.0-1.0) |
| `--no-auto-detect`           | Skip HuggingFace model auto-detection     |

### Registry commands

| Command                                              | Description                                           |
|------------------------------------------------------|-------------------------------------------------------|
| `sparkrun registry list`                             | List configured recipe registries                     |
| `sparkrun registry add <url>`                        | Add a registry from a repo's manifest                 |
| `sparkrun registry remove <name>`                    | Remove a recipe registry                              |
| `sparkrun registry enable <name>`                    | Enable a disabled registry                            |
| `sparkrun registry disable <name>`                   | Disable a registry (keeps config)                     |
| `sparkrun registry update [name]`                    | Update registries from git (optionally just one)      |
| `sparkrun registry list-benchmark-profiles`          | List available benchmark profiles                     |
| `sparkrun registry show-benchmark-profile <name>`    | Show benchmark profile details                        |

### Cluster commands

| Command                               | Description                                         |
|---------------------------------------|-----------------------------------------------------|
| `sparkrun cluster create <name>`      | Create a new named cluster (`--user` sets SSH user) |
| `sparkrun cluster update <name>`      | Update hosts, description, or user of a cluster     |
| `sparkrun cluster list`               | List all saved clusters                             |
| `sparkrun cluster show <name>`        | Show details of a saved cluster                     |
| `sparkrun cluster delete <name>`      | Delete a saved cluster                              |
| `sparkrun cluster set-default <name>` | Set the default cluster                             |
| `sparkrun cluster unset-default`      | Remove the default cluster setting                  |
| `sparkrun cluster default`            | Show the current default cluster                    |
| `sparkrun cluster status`             | Show running containers, pending operations, and IP mappings |
| `sparkrun status`                     | Alias for `sparkrun cluster status`                          |

The first host in a cluster definition is used as the **head node** for multi-node jobs. Order the remaining
hosts however you like — they become workers.

### Setup commands

| Command                            | Description                                       |
|------------------------------------|---------------------------------------------------|
| `sparkrun setup install`           | Install sparkrun as a uv tool + tab-completion    |
| `sparkrun setup completion`        | Install shell tab-completion (bash/zsh/fish)      |
| `sparkrun setup update`            | Update sparkrun + registries (`--no-update-registries` to skip) |
| `sparkrun setup ssh`               | Set up passwordless SSH mesh across hosts         |
| `sparkrun setup cx7`               | Detect and configure ConnectX-7 NICs across hosts |
| `sparkrun setup fix-permissions`   | Fix root-owned HF cache files on cluster hosts    |
| `sparkrun setup clear-cache`       | Drop Linux page cache on cluster hosts             |

Both `setup fix-permissions` and `setup clear-cache` accept `--save-sudo` to install a scoped sudoers entry
so future runs don't require password prompts.

## Roadmap

- Additional bundled recipes for popular models
- Health checks and status monitoring for running workloads

## About

sparkrun provides a unified tool for running inference on DGX Spark systems without Slurm or Kubernetes coordination. It
is intended to be donated to a future community organization.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
