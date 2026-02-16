"""Single-node launcher for sparkrun.

Handles launching inference workloads on a single DGX Spark host.
"""

from __future__ import annotations

import logging
import subprocess
import time

from sparkrun.config import SparkrunConfig, DEFAULT_HF_CACHE_DIR
from sparkrun.orchestration.ssh import RemoteResult, run_remote_script, run_remote_command
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
)
from sparkrun.orchestration.scripts import (
    generate_container_launch_script,
    generate_exec_serve_script,
)
from sparkrun.orchestration.docker import docker_stop_cmd, generate_container_name

logger = logging.getLogger(__name__)


def run_solo(
    host: str,
    image: str,
    serve_command: str,
    cluster_id: str = "sparkrun0",
    env: dict[str, str] | None = None,
    cache_dir: str | None = None,
    config: SparkrunConfig | None = None,
    dry_run: bool = False,
    detached: bool = True,
    skip_ib_detect: bool = False,
) -> int:
    """Launch a single-node inference workload.

    Steps:

    1. Detect InfiniBand on the target host (optional).
    2. Launch container with ``sleep infinity`` to keep it alive.
    3. Execute the serve command inside the container.

    Args:
        host: Target hostname or IP (use ``"localhost"`` for local execution).
        image: Container image to use.
        serve_command: The inference serve command to run.
        cluster_id: Identifier for container naming.
        env: Additional environment variables.
        cache_dir: HuggingFace cache directory path.
        config: SparkrunConfig instance for SSH settings.
        dry_run: If True, show what would be done without executing.
        detached: If True, run serve command in background (survives SSH disconnect).
        skip_ib_detect: Skip InfiniBand detection.

    Returns:
        Exit code (0 = success).
    """
    is_local = host in ("localhost", "127.0.0.1", "")
    container_name = generate_container_name(cluster_id, "solo")
    hf_cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    volumes = {hf_cache: "/root/.cache/huggingface"}

    ssh_kwargs: dict = {}
    if config:
        ssh_kwargs["ssh_user"] = config.ssh_user
        ssh_kwargs["ssh_key"] = config.ssh_key
        ssh_kwargs["ssh_options"] = config.ssh_options

    all_env = dict(env or {})

    # Step 1: InfiniBand detection
    t0 = time.monotonic()
    nccl_env: dict[str, str] = {}
    if not skip_ib_detect:
        logger.info("Step 1/3: Detecting InfiniBand on %s...", host)
        ib_script = generate_ib_detect_script()
        if is_local:
            result = _run_local_script(ib_script, dry_run=dry_run)
        else:
            result = run_remote_script(host, ib_script, timeout=30, dry_run=dry_run, **ssh_kwargs)

        if result.success:
            ib_info = parse_ib_detect_output(result.stdout)
            nccl_env = generate_nccl_env(ib_info)
            if nccl_env:
                logger.info("  InfiniBand detected, NCCL configured")
            else:
                logger.info("  No InfiniBand detected, using default networking")
        else:
            logger.warning(
                "  InfiniBand detection failed, continuing without: %s",
                result.stderr[:100],
            )
        logger.info("Step 1/3: IB detection done (%.1fs)", time.monotonic() - t0)
    else:
        logger.info("Step 1/3: Skipping InfiniBand detection")

    # Step 2: Launch container
    t0 = time.monotonic()
    logger.info("Step 2/3: Launching container %s on %s (image: %s)...", container_name, host, image)
    launch_script = generate_container_launch_script(
        image=image,
        container_name=container_name,
        command="sleep infinity",  # keep container alive for exec
        env=all_env,
        volumes=volumes,
        nccl_env=nccl_env,
    )

    if is_local:
        result = _run_local_script(launch_script, dry_run=dry_run)
    else:
        result = run_remote_script(host, launch_script, timeout=120, dry_run=dry_run, **ssh_kwargs)

    if not result.success and not dry_run:
        logger.error("Failed to launch container: %s", result.stderr)
        return 1
    logger.info("Step 2/3: Container launched (%.1fs)", time.monotonic() - t0)

    # Step 3: Execute serve command
    t0 = time.monotonic()
    logger.info("Step 3/3: Executing serve command in %s...", container_name)
    exec_script = generate_exec_serve_script(
        container_name=container_name,
        serve_command=serve_command,
        env=all_env,
        detached=detached,
    )

    if is_local:
        result = _run_local_script(exec_script, dry_run=dry_run)
    else:
        result = run_remote_script(host, exec_script, timeout=60, dry_run=dry_run, **ssh_kwargs)
    logger.info("Step 3/3: Serve command dispatched (%.1fs)", time.monotonic() - t0)

    if dry_run:
        return 0

    return result.returncode


def stop_solo(
    host: str,
    cluster_id: str = "sparkrun0",
    config: SparkrunConfig | None = None,
    dry_run: bool = False,
) -> int:
    """Stop a running solo workload.

    Args:
        host: Target hostname or IP.
        cluster_id: Cluster identifier used when launching.
        config: SparkrunConfig instance for SSH settings.
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    container_name = generate_container_name(cluster_id, "solo")
    stop_cmd = docker_stop_cmd(container_name)
    is_local = host in ("localhost", "127.0.0.1", "")

    ssh_kwargs: dict = {}
    if config:
        ssh_kwargs["ssh_user"] = config.ssh_user
        ssh_kwargs["ssh_key"] = config.ssh_key
        ssh_kwargs["ssh_options"] = config.ssh_options

    if is_local:
        result = _run_local_script(f"#!/bin/bash\n{stop_cmd}", dry_run=dry_run)
    else:
        result = run_remote_command(host, stop_cmd, dry_run=dry_run, **ssh_kwargs)

    return 0 if result.success else 1


def _run_local_script(script: str, dry_run: bool = False) -> RemoteResult:
    """Execute a script locally via subprocess.

    Args:
        script: Bash script content to execute.
        dry_run: If True, log the script but don't execute.

    Returns:
        RemoteResult with host set to ``"localhost"``.
    """
    if dry_run:
        logger.info("[dry-run] Would execute locally:\n%s", script)
        return RemoteResult(host="localhost", returncode=0, stdout="[dry-run]", stderr="")

    proc = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
    )
    return RemoteResult(
        host="localhost",
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
