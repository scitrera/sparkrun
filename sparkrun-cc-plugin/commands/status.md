# /sparkrun:status

Check the status of running sparkrun inference workloads.

## Usage

```
/sparkrun:status [options]
```

## Examples

```
/sparkrun:status
/sparkrun:status --cluster mylab
/sparkrun:status --hosts 192.168.11.13,192.168.11.14
```

## Behavior

When this command is invoked:

1. Run `sparkrun status` with appropriate host resolution:

```bash
sparkrun status --cluster <name>
# or
sparkrun status --hosts <ip1>,<ip2>,...
# or just (uses default cluster)
sparkrun status
```

2. Report the output to the user. The status command shows:
   - Grouped containers by job (with recipe name if cached)
   - Container role, host, status, and image
   - Ready-to-use `sparkrun logs` and `sparkrun stop` commands for each job

3. If the user wants to act on a specific job (view logs, stop it), use the commands shown in the status output.

## Notes

- This is equivalent to `sparkrun cluster status`
- Uses the default cluster if no `--hosts` or `--cluster` is specified
- Container names follow the pattern `sparkrun_{hash}_{role}`
