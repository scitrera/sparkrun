# sparkrun Plugin for Claude Code

AI-assisted inference on NVIDIA DGX Spark -- run, manage, and stop LLM workloads with Claude.

## What It Does

This plugin teaches Claude Code how to use [sparkrun](https://github.com/scitrera/sparkrun) to manage LLM inference workloads on NVIDIA DGX Spark systems. It provides:

- **Slash Commands** -- Quick actions for running, stopping, and managing inference jobs
- **Skills** -- Detailed reference that Claude uses automatically when working with sparkrun

## Installation

### From the Marketplace

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add scitrera/sparkrun

# Install the plugin
claude plugin install sparkrun@sparkrun
```

### Manual Installation

Clone or copy the plugin directory:

```bash
# Global (available in all projects)
cp -r sparkrun-cc-plugin ~/.claude/plugins/sparkrun

# Or project-local
cp -r sparkrun-cc-plugin .claude/plugins/sparkrun
```

### Local Development

```bash
claude --plugin-dir ./sparkrun-cc-plugin
```

## Prerequisites

### sparkrun CLI

The plugin requires sparkrun to be installed:

```bash
# Install via uvx (recommended)
uvx sparkrun setup install

# Or via uv
uv tool install sparkrun
```

### DGX Spark Cluster

You need SSH access to one or more NVIDIA DGX Spark systems. Set up a cluster config:

```bash
sparkrun cluster create mylab --hosts 192.168.11.13,192.168.11.14 -d "My DGX Spark lab"
sparkrun cluster set-default mylab
```

For multi-node inference, set up passwordless SSH:

```bash
sparkrun setup ssh --cluster mylab
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/sparkrun:run <recipe>` | Launch an inference workload |
| `/sparkrun:stop <recipe>` | Stop a running workload |
| `/sparkrun:status` | Check status of running workloads |
| `/sparkrun:list [query]` | Browse and search available recipes |
| `/sparkrun:setup` | Guided setup for sparkrun and cluster config |

## Skills (Automatic)

Claude automatically uses these skills when the task context matches:

| Skill | Activates When |
|-------|----------------|
| `run` | Running, monitoring, or stopping inference workloads |
| `setup` | Installing sparkrun, configuring clusters, SSH setup |
| `registry` | Managing recipe registries, creating/editing recipes |

## Usage Examples

You can use the slash commands directly:

```
/sparkrun:run qwen3-1.7b-vllm --tp 2
/sparkrun:status
/sparkrun:list qwen3
```

Or just describe what you want in natural language -- Claude will use the skills automatically:

- "Run the Qwen3 1.7B model on my cluster"
- "What inference jobs are running?"
- "Stop the nemotron model"
- "Show me available recipes for llama models"
- "Set up sparkrun on my DGX Spark cluster"
- "Create a recipe for Mistral 7B on vLLM"

## Key Concepts

- **Recipes** are YAML files describing an inference workload (model, runtime, container, defaults)
- **Runtimes** are inference engines: vLLM, SGLang, llama.cpp
- **Clusters** are named groups of DGX Spark hosts
- Each DGX Spark has 1 GPU, so `--tp N` (tensor parallelism) = N hosts
- sparkrun launches detached containers -- Ctrl+C detaches from logs, never kills the job

## Links

- [sparkrun Documentation](https://github.com/scitrera/sparkrun)
- [Recipe Format Specification](https://github.com/scitrera/sparkrun/blob/main/RECIPES.md)

## License

Apache 2.0 License -- see [LICENSE](../LICENSE) for details.
