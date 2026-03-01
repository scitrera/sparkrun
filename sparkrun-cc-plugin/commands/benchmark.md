# /sparkrun:benchmark

Run benchmarks against an inference workload.

## Usage

```
/sparkrun:benchmark <recipe> [options]
```

## Examples

```
/sparkrun:benchmark qwen3-1.7b-sglang --solo
/sparkrun:benchmark qwen3-1.7b-sglang --tp 2 --profile spark-arena-v1
/sparkrun:benchmark qwen3-1.7b-sglang --skip-run --solo
/sparkrun:benchmark qwen3-1.7b-sglang --no-stop --cluster mylab
```

## Behavior

When this command is invoked:

1. If no recipe is specified, run `sparkrun list` to show available recipes and ask the user to pick one.
2. Determine the target hosts (same as `/sparkrun:run`).
3. Run the full benchmark flow:

```bash
# Full flow: launch inference -> benchmark -> stop
sparkrun benchmark <recipe> [options]
```

The benchmark command handles the complete lifecycle:
- **Step 1/3**: Launches the inference server (unless `--skip-run`)
- **Step 2/3**: Runs the benchmark against the server
- **Step 3/3**: Stops the inference server (unless `--no-stop`)

4. Results are saved to a YAML file (and JSON if available).

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--solo` | Force single-host mode |
| `--tp N` | Tensor parallelism (= number of nodes) |
| `--port N` | Override serve port |
| `--image` | Override container image |
| `--cache-dir` | HuggingFace cache directory |
| `--profile` | Benchmark profile name or file path |
| `--framework` | Override benchmarking framework (default: llama-benchy) |
| `-o KEY=VALUE` | Override benchmark arg (repeatable) |
| `--no-stop` | Keep inference running after benchmarking |
| `--skip-run` | Skip launching inference (benchmark existing instance) |
| `--timeout` | Benchmark timeout in seconds (default: 14400) |
| `--out, --output` | Output file for results YAML |
| `--dry-run` | Show what would be done |

## Benchmark Profiles

Browse available profiles from registries:

```bash
sparkrun registry list-benchmark-profiles
sparkrun registry show-benchmark-profile <name>
```

## Notes

- The benchmark command auto-detects available ports to avoid collisions with running instances
- Use `--skip-run` to benchmark an already-running inference server
- Use `--no-stop` to keep the inference server running after the benchmark completes
- Benchmark args from `--profile` can be overridden with `-o key=value`
- Results are saved as both YAML and JSON when available
