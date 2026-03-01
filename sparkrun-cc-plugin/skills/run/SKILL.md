---
name: run
description: "ALWAYS invoke this skill before running any sparkrun CLI commands. Never run sparkrun directly via Bash without loading this skill first. Covers launching, monitoring, stopping, and checking status of inference workloads on NVIDIA DGX Spark."
---

<Purpose>
Provides complete reference for launching, monitoring, and stopping LLM inference workloads using sparkrun on NVIDIA DGX Spark systems. Covers the full lifecycle: browse recipes, check VRAM fit, launch jobs, view logs, check status, stop workloads, and run benchmarks.
</Purpose>

<Use_When>
- User wants to run an LLM inference model on DGX Spark
- User wants to check status of running workloads
- User wants to stop a running inference job
- User wants to view logs from a running workload
- User wants to preview VRAM requirements before launching
- User wants to benchmark an inference workload
- User asks "how do I run", "start", "launch", "deploy" a model
</Use_When>

<Do_Not_Use_When>
- User wants to install sparkrun or set up a cluster -- use the setup skill instead
- User wants to manage recipe registries or create custom recipes -- use the registry skill instead
- User is asking about sparkrun internals or development
</Do_Not_Use_When>

<Steps>

## Run a Recipe

```bash
# Solo (single host)
sparkrun run <recipe> --solo --no-follow
sparkrun run <recipe> --hosts <ip> --no-follow

# Multi-node cluster
sparkrun run <recipe> --cluster <name> --no-follow
sparkrun run <recipe> --hosts <ip1>,<ip2>,... --no-follow
sparkrun run <recipe> --tp <N> --no-follow

# With overrides
sparkrun run <recipe> --port 9000 --gpu-mem 0.8 --no-follow
sparkrun run <recipe> -o max_model_len=8192 -o attention_backend=triton --no-follow

# Dry-run (show what would happen)
sparkrun run <recipe> --dry-run
```

**CRITICAL: Always use `--no-follow`** when running from an agent/skill context to avoid blocking on log streaming. Then use `sparkrun cluster status` or `sparkrun logs` separately to check on the job.

## Check Status

```bash
# Show all sparkrun containers across cluster hosts
# NOTE: `sparkrun status` is an alias for `sparkrun cluster status` — only run one.
sparkrun cluster status

# With explicit targets
sparkrun cluster status --cluster <name>
sparkrun cluster status --hosts <ip1>,<ip2>,...

# Output shows:
#   - Grouped containers by job (with recipe name if cached)
#   - Container role, host, status, and image
#   - Ready-to-use logs and stop commands for each job
```

## View Logs

```bash
sparkrun logs <recipe> --cluster <name>
sparkrun logs <recipe> --hosts <ip1>,<ip2>,...
sparkrun logs <recipe> --tp <N>
```

## Stop a Workload

```bash
sparkrun stop <recipe> --cluster <name>
sparkrun stop <recipe> --hosts <ip1>,<ip2>,...
sparkrun stop <recipe> --tp <N>
sparkrun stop <recipe> --dry-run

# Stop all sparkrun containers (no recipe needed)
sparkrun stop --all --cluster <name>
sparkrun stop --all --hosts <ip1>,<ip2>,...
```

## Browse and Inspect Recipes

```bash
# List all available recipes (no filter)
sparkrun list

# List with filters
sparkrun list --all                         # include hidden registry recipes
sparkrun list --registry <name>             # filter by registry
sparkrun list --runtime vllm                # filter by runtime

# Search for recipes by name, model, runtime, or description (contains-match)
sparkrun recipe search <query>
sparkrun recipe search <query> --registry <name> --runtime sglang

# Inspect a specific known recipe (by exact name or file path)
sparkrun recipe show <recipe>
sparkrun recipe show <recipe> --tp <N>

# Validate or estimate VRAM
sparkrun recipe validate <recipe>
sparkrun recipe vram <recipe> --tp <N> --max-model-len 32768
```

Use `sparkrun recipe search` as the first attempt when looking for a particular recipe. Use `sparkrun recipe show` when given a specific recipe name or file -- it may not appear in search results.

## Benchmark

```bash
# Full flow: launch inference -> benchmark -> stop
sparkrun benchmark <recipe> --solo
sparkrun benchmark <recipe> --cluster <name>
sparkrun benchmark <recipe> --tp 2 --profile <profile_name>

# Benchmark an already-running instance (skip launch)
sparkrun benchmark <recipe> --skip-run --solo

# Keep inference running after benchmark completes
sparkrun benchmark <recipe> --no-stop --solo

# Override benchmark args
sparkrun benchmark <recipe> -o depth=0,2048,4096 -o tg=32,128

# Specify framework and timeout
sparkrun benchmark <recipe> --framework llama-benchy --timeout 3600

# Dry-run
sparkrun benchmark <recipe> --dry-run
```

</Steps>

<Tool_Usage>
All sparkrun commands are executed via the Bash tool. No MCP tools are required.

When running workloads:
1. Always use `--no-follow` flag with `sparkrun run`
2. After launching, run `sparkrun cluster status` to confirm containers are running
3. Use the logs/stop commands from status output to manage jobs
</Tool_Usage>

<Key_Options>

**`sparkrun run` options:**

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--hosts-file` | File with hosts (one per line) |
| `--cluster` | Use a saved cluster |
| `--solo` | Force single-host mode |
| `--tp, --tensor-parallel` | Override tensor parallelism (= node count) |
| `--port` | Override serve port |
| `--gpu-mem` | GPU memory utilization (0.0-1.0) |
| `--image` | Override container image |
| `--cache-dir` | HuggingFace cache directory |
| `-o KEY=VALUE` | Override any recipe default |
| `--ray-port` | Ray GCS port for vllm-ray (default: 46379) |
| `--init-port` | vllm/SGLang distributed init port (default: 25000) |
| `--dashboard` | Enable Ray dashboard on head node |
| `--dashboard-port` | Ray dashboard port (default: 8265) |
| `--dry-run, -n` | Show what would be done |
| `--no-follow` | Don't attach to logs after launch |
| `--foreground` | Run in foreground (blocking) |

</Key_Options>

<Important_Notes>
- **Always use `--no-follow`** when running from an automated/agent context to avoid blocking
- Use `sparkrun cluster status` after launching to confirm containers are running
- `--tp N` must match the number of hosts (DGX Spark = 1 GPU per host)
- `sparkrun stop` and `sparkrun logs` need the same `--hosts`/`--cluster`/`--tp` flags as the original `run`
- Use `sparkrun show <recipe> --tp N` to preview VRAM estimates before running
- Container names follow the pattern `sparkrun_{hash}_{role}` where the hash is derived from runtime + model + sorted hosts
- Ctrl+C while following logs detaches safely -- it never kills the inference job
- Use `sparkrun stop --all` to stop all sparkrun containers without specifying a recipe
</Important_Notes>

Task: {{ARGUMENTS}}
