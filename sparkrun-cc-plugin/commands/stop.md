# /sparkrun:stop

Stop a running inference workload.

## Usage

```
/sparkrun:stop <recipe> [options]
```

## Examples

```
/sparkrun:stop qwen3-1.7b-vllm
/sparkrun:stop nemotron3-nano-30b-nvfp4-vllm --tp 2
/sparkrun:stop nemotron3-nano-30b-nvfp4-vllm --cluster mylab
```

## Behavior

When this command is invoked:

1. If no recipe is specified, run `sparkrun status` first to see what's running and ask the user which job to stop.
2. Stop the workload:

```bash
sparkrun stop <recipe> [options]
```

3. The `--hosts`/`--cluster`/`--tp` flags must match the original `run` command so sparkrun can find the right containers.
4. After stopping, optionally run `sparkrun status` to confirm containers are gone.

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--tp N` | Must match the value used in `run` |
| `--dry-run` | Show what would be done |

## Notes

- The stop command identifies containers by a hash derived from runtime + model + sorted hosts
- If `--tp` was used during `run`, it must also be passed to `stop`
- Use `sparkrun status` to see the exact stop commands for running jobs
