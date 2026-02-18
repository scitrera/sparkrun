# /sparkrun:run

Run an inference workload on DGX Spark using a sparkrun recipe.

## Usage

```
/sparkrun:run <recipe> [options]
```

## Examples

```
/sparkrun:run qwen3-1.7b-vllm
/sparkrun:run nemotron3-nano-30b-nvfp4-vllm --tp 2
/sparkrun:run qwen3-1.7b-llama-cpp --solo
```

## Behavior

When this command is invoked:

1. If no recipe is specified, run `sparkrun list` to show available recipes and ask the user to pick one.
2. Determine the target hosts:
   - If the user specifies `--hosts`, `--cluster`, or `--solo`, use those.
   - Otherwise check if a default cluster is configured (`sparkrun cluster default`).
   - If no hosts can be resolved, ask the user for hosts or to create a cluster first.
3. Optionally run `sparkrun show <recipe> --tp <N>` to preview VRAM estimation before launching.
4. Run the workload:

```bash
sparkrun run <recipe> [options] --no-follow
```

**CRITICAL: Always use `--no-follow`** to avoid blocking on log streaming. After launch, use `sparkrun status` or `sparkrun logs` separately.

5. After launching, run `sparkrun status` to confirm containers are running.

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--solo` | Force single-host mode |
| `--tp N` | Tensor parallelism (= number of nodes) |
| `--port N` | Override serve port |
| `--gpu-mem F` | GPU memory utilization (0.0-1.0) |
| `-o KEY=VALUE` | Override any recipe default |
| `--dry-run` | Show what would be done |

## Notes

- Each DGX Spark has 1 GPU, so `--tp N` means N hosts
- `sparkrun stop` and `sparkrun logs` need the same `--hosts`/`--cluster`/`--tp` flags as `run`
- Ctrl+C while following logs detaches safely — it never kills the inference job
