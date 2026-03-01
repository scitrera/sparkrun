# /sparkrun:setup

Install sparkrun and configure a DGX Spark cluster.

## Usage

```
/sparkrun:setup
```

## Behavior

When this command is invoked, walk the user through setup:

### Step 1: Check if sparkrun is installed

```bash
which sparkrun && sparkrun --version
```

If not installed, install it:

```bash
uvx sparkrun setup install
```

This creates a managed virtual environment, installs sparkrun, and sets up shell tab-completion.

### Step 2: Configure a cluster

Ask the user for their DGX Spark host IPs, then create and set a default cluster:

```bash
sparkrun cluster create <name> --hosts <ip1>,<ip2>,... -d "<description>" --user <ssh_user>
sparkrun cluster set-default <name>
```

### Step 3: Set up SSH mesh

Multi-node inference requires passwordless SSH. Run the mesh setup interactively:

```bash
sparkrun setup ssh --cluster <name>
```

**IMPORTANT:** This command is interactive (prompts for passwords). Do NOT capture its output. Let it pass through to the terminal.

### Step 4: Configure CX7 networking (optional)

If the cluster has ConnectX-7 interfaces for high-speed transfers:

```bash
sparkrun setup cx7 --cluster <name>
```

This detects CX7 interfaces, assigns static IPs, and applies netplan configuration. Requires sudo.

### Step 5: Fix file permissions (optional)

If Docker containers have created root-owned files in the HuggingFace cache:

```bash
sparkrun setup fix-permissions --cluster <name>
```

Optionally install a sudoers entry for passwordless future runs:

```bash
sparkrun setup fix-permissions --cluster <name> --save-sudo
```

### Step 6: Clear page cache (optional)

Free cached file data to maximize available memory for inference:

```bash
sparkrun setup clear-cache --cluster <name>
```

### Step 7: Verify

Run a quick check to confirm everything works:

```bash
sparkrun cluster show <name>
sparkrun list
sparkrun show <recipe> --tp <N>
```

### Step 8: Summary

Report the setup results:
- sparkrun version
- Cluster name, hosts, default status
- SSH mesh status
- Available recipe count

## Notes

- `uv` is the recommended Python package manager; install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- The first host in a cluster is the head node for multi-node jobs
- DGX Spark has 1 GPU per host, so tensor_parallel = number of hosts
- SSH user defaults to current OS user; override with `--user`
- Use `sparkrun setup update` to update sparkrun and registries to the latest version
