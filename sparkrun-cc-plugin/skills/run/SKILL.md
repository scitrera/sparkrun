---
name: run
description: Run, monitor, and stop inference workloads on NVIDIA DGX Spark
---

<Purpose>
Provides complete reference for launching, monitoring, and stopping LLM inference workloads using sparkrun on NVIDIA DGX Spark systems. Covers the full lifecycle: browse recipes, check VRAM fit, launch jobs, view logs, check status, and stop workloads.
</Purpose>

<Use_When>
- User wants to run an LLM inference model on DGX Spark
- User wants to check status of running workloads
- User wants to stop a running inference job
- User wants to view logs from a running workload
- User wants to preview VRAM requirements before launching
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

**CRITICAL: Always use `--no-follow`** when running from an agent/skill context to avoid blocking on log streaming. Then use `sparkrun status` or `sparkrun logs` separately to check on the job.

## Check Status

```bash
# Show all sparkrun containers across cluster hosts
sparkrun status --cluster <name>
sparkrun status --hosts <ip1>,<ip2>,...
sparkrun status

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
```

## Browse and Inspect Recipes

```bash
sparkrun list
sparkrun list <query>
sparkrun show <recipe>
sparkrun show <recipe> --tp <N>
sparkrun search <query>
sparkrun recipe validate <recipe>
sparkrun recipe vram <recipe> --tp <N> --max-model-len 32768
```

</Steps>

<Tool_Usage>
All sparkrun commands are executed via the Bash tool. No MCP tools are required.

When running workloads:
1. Always use `--no-follow` flag with `sparkrun run`
2. After launching, run `sparkrun status` to confirm containers are running
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
| `-o KEY=VALUE` | Override any recipe default |
| `--dry-run, -n` | Show what would be done |
| `--no-follow` | Don't attach to logs after launch |
| `--foreground` | Run in foreground (blocking) |
| `--skip-ib` | Skip InfiniBand detection |

</Key_Options>

<Important_Notes>
- **Always use `--no-follow`** when running from an automated/agent context to avoid blocking
- Use `sparkrun status` after launching to confirm containers are running
- `--tp N` must match the number of hosts (DGX Spark = 1 GPU per host)
- `sparkrun stop` and `sparkrun logs` need the same `--hosts`/`--cluster`/`--tp` flags as the original `run`
- Use `sparkrun show <recipe> --tp N` to preview VRAM estimates before running
- Container names follow the pattern `sparkrun_{hash}_{role}` where the hash is derived from runtime + model + sorted hosts
- Ctrl+C while following logs detaches safely -- it never kills the inference job
</Important_Notes>

Task: {{ARGUMENTS}}
