# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sparkrun** is a CLI tool for launching, managing, and stopping Docker-based LLM inference workloads on NVIDIA DGX Spark systems. It orchestrates containers over SSH — no Slurm or Kubernetes required. The control machine doesn't need to be a cluster member; it coordinates DGX Sparks remotely.

Each DGX Spark has one GPU with 128 GB unified memory, so tensor parallelism maps directly to node count (`--tp 2` = 2 hosts).

## Common Commands

```bash
# Install in development mode (editable)
uv pip install -e ".[dev]"

# Run full test suite
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_recipe.py -v

# Run a specific test
.venv/bin/python -m pytest tests/test_cli.py::test_run_command_basic -v

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=sparkrun --cov-report=term-missing

# Lint (ruff, line-length 120, target py312)
ruff check src/ tests/
ruff format src/ tests/

# Run the CLI directly during development
.venv/bin/sparkrun --help
.venv/bin/sparkrun run --dry-run qwen3-1.7b-vllm

# Sync versions across packages (pyproject.toml + sparkrun-cc-plugin)
python scripts/update-versions.py
python scripts/update-versions.py --check   # CI-friendly verify
```

Versions are tracked in `versions.yaml` at the repo root and synced to all package files via `scripts/update-versions.py`.

## Architecture

### Source Layout

```
src/sparkrun/
├── cli.py              # Click CLI — all user-facing commands (~74KB, the largest file)
├── bootstrap.py        # SAF plugin initialization, runtime discovery
├── config.py           # SparkrunConfig — reads ~/.config/sparkrun/config.yaml
├── cluster_manager.py  # Named cluster CRUD (YAML files in ~/.config/sparkrun/clusters/)
├── hosts.py            # Host resolution priority chain (CLI → file → cluster → default)
├── recipe.py           # Recipe loading, validation, v1→v2 migration, config chain
├── registry.py         # Git-based recipe registry system (sparse checkouts)
├── vram.py             # VRAM estimation with HuggingFace model auto-detection
├── pending_ops.py      # PID-based lock files for in-progress operations
├── runtimes/           # Runtime plugins (see below)
├── orchestration/      # SSH, Docker, InfiniBand, script execution primitives
├── models/             # HuggingFace model download and distribution
├── containers/         # Container image distribution (docker save/load over SSH)
└── scripts/            # Embedded bash scripts (IB detection, container launch, etc.)
```

### Plugin System (SAF)

sparkrun uses [scitrera-app-framework](https://github.com/scitrera/python-app-framework) (SAF) for plugin discovery and lifecycle. Runtimes register as multi-extension plugins under the `sparkrun.runtime` extension point. Discovery happens via Python entry points defined in `pyproject.toml` (`[project.entry-points."sparkrun.runtimes"]`).

Key bootstrap flow: `cli.py` → `bootstrap.init_sparkrun()` → SAF `init_framework_desktop()` → `find_types_in_modules("sparkrun.runtimes", RuntimePlugin)` → `register_plugin()` for each discovered runtime.

### Runtime Architecture

All runtimes extend `RuntimePlugin` (in `runtimes/base.py`), which itself extends SAF's `Plugin` class. The base class provides solo-mode orchestration; runtimes override `run()`/`stop()`/`follow_logs()` for multi-node support.

| Runtime | File | Clustering | Strategy |
|---------|------|-----------|----------|
| **vllm** | `runtimes/vllm.py` | Ray head/worker | `"ray"` — starts Ray cluster, exec serve on head |
| **sglang** | `runtimes/sglang.py` | Native distributed | `"native"` — each node runs serve with `--node-rank` |
| **llama-cpp** | `runtimes/llama_cpp.py` | Experimental RPC | `"rpc"` — workers run `rpc-server`, head connects via `--rpc` |
| **eugr-vllm** | `runtimes/eugr_vllm.py` | Ray (inherited) | Extends VllmRuntime with eugr container builds and mods |

Runtimes must implement `generate_command()` and `resolve_container()`. The `cluster_strategy()` return value determines which orchestration path the base class uses.

### Orchestration Layer (`orchestration/`)

All remote operations use **SSH stdin piping** — scripts are generated as Python strings and piped to `ssh <host> bash -s`. No files are ever copied to remote hosts.

- **`ssh.py`** — `run_remote_script()`, `run_remote_scripts_parallel()`, `run_remote_sudo_script()`, `run_rsync_parallel()`, `detect_sudo_on_hosts()`, `stream_remote_logs()`
- **`docker.py`** — Pure command-string generators (`docker_run_cmd`, `docker_exec_cmd`, etc.), cluster ID generation
- **`distribution.py`** — High-level resource distribution: IB detection, container image and model syncing to target hosts (orchestrates `models/`, `containers/`, and IB detection)
- **`infiniband.py`** — IB detection script generation, NCCL env var computation, IB IP mapping for fast transfers
- **`networking.py`** — ConnectX-7 NIC detection, IP assignment planning, CX7 configuration script generation, host key distribution
- **`primitives.py`** — Higher-level composition: `build_ssh_kwargs()`, `build_volumes()`, `merge_env()`, `detect_infiniband()`, `run_script_on_host()`, `cleanup_containers()`
- **`job_metadata.py`** — Persistent job metadata (cluster_id → recipe mapping) stored in `~/.cache/sparkrun/jobs/`
- **`scripts.py`** — Script file loader (`read_script()`) for embedded bash scripts in `scripts/`

### Recipe System

Recipes are YAML files with fields: `model`, `runtime`, `container`, `command`, `defaults`, `env`, `metadata`, `min_nodes`, `max_nodes`. The `Recipe` class uses VPD (`vpd` library) for config chain resolution — CLI overrides → recipe defaults → runtime defaults.

Recipe resolution: `cli.py` → `RegistryManager.find_recipe()` → searches bundled recipes, local `./recipes/`, user config recipes, and git-cloned registries.

Two recipe format versions exist: v1 (eugr-style, auto-detected by `recipe_version: "1"` or presence of `build_args`/`mods`) and v2 (sparkrun native). See `RECIPES.md` for the full specification.

### Model & Container Distribution

Before launching, sparkrun can pre-sync models and container images from the control machine to target hosts:

- **Models** (`models/`): Downloads from HuggingFace Hub locally via `snapshot_download`, then rsyncs to targets. GGUF models use colon syntax (`repo:quant`) for selective quant-file download.
- **Containers** (`containers/`): Pulls image locally, then streams via `docker save | ssh docker load`. Checks image IDs to skip hosts that already have the correct image.

### Config & State Paths

| Path | Purpose |
|------|---------|
| `~/.config/sparkrun/config.yaml` | User configuration |
| `~/.config/sparkrun/clusters/*.yaml` | Named cluster definitions |
| `~/.config/sparkrun/registries.yaml` | Custom recipe registry list |
| `~/.cache/sparkrun/registries/` | Git-cloned recipe registries |
| `~/.cache/sparkrun/jobs/` | Job metadata (cluster_id → recipe mapping) |
| `~/.cache/sparkrun/pending/` | PID lock files for in-progress operations |
| `~/.cache/huggingface/` | HuggingFace model cache (mounted into containers) |

### Testing Patterns

Tests use pytest with `pytest-asyncio`. The `conftest.py` provides an `isolate_stateful` autouse fixture that redirects SAF's stateful root to `tmp_path`, preventing tests from touching `~/.config/sparkrun/`. The bootstrap singleton (`_variables`) is reset between tests.

All SSH/Docker operations in tests are mocked — no real hosts are needed. Common fixtures: `tmp_recipe_dir` (creates sample v1/v2 recipes), `cluster_dir`, `hosts_file`, `v` (initialized SAF Variables instance).

### Companion Packages

- **`sparkrun-cc-plugin/`** — Claude Code plugin providing slash commands (`/sparkrun:run`, `/sparkrun:stop`, `/sparkrun:status`, `/sparkrun:list`, `/sparkrun:setup`) and automatic skills for AI-assisted inference management.
- **`website/`** — Documentation site built with Astro (Starlight theme), deployed to Cloudflare Pages.

## Key Dependencies

- **`scitrera-app-framework`** (SAF) — Plugin system, lifecycle, variables/config management
- **`vpd`** — Virtual Path Dict for YAML reading and config chain resolution (`vpd_chain`, `arg_substitute`)
- **`click`** — CLI framework
- **`huggingface_hub`** — Model downloading (`snapshot_download`)
- **`pyyaml`** — YAML parsing for recipes, clusters, registries
