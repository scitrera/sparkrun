"""Reusable orchestration primitives for sparkrun.

Higher-level building blocks composed from the low-level modules
(ssh, docker, infiniband, scripts).  Runtimes use these to assemble
their particular launch and teardown flows.
"""

from __future__ import annotations

import logging
import subprocess
import time

from sparkrun.config import SparkrunConfig, DEFAULT_HF_CACHE_DIR
from sparkrun.orchestration.ssh import (
    RemoteResult,
    run_remote_command,
    run_remote_script,
    run_remote_scripts_parallel,
)
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
)
from sparkrun.orchestration.docker import docker_stop_cmd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_ssh_kwargs(config: SparkrunConfig | None) -> dict:
    """Extract SSH connection parameters from a SparkrunConfig.

    Returns a dict suitable for ``**kwargs`` into :func:`run_remote_script`
    and friends.
    """
    if not config:
        return {}
    return {
        "ssh_user": config.ssh_user,
        "ssh_key": config.ssh_key,
        "ssh_options": config.ssh_options,
    }


def build_volumes(
        cache_dir: str | None = None,
        extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the standard volume mapping for HuggingFace cache + extras.

    Args:
        cache_dir: Host-side HF cache path (defaults to
            :data:`sparkrun.config.DEFAULT_HF_CACHE_DIR`).
        extra: Additional host→container volume mappings.

    Returns:
        Merged volume dict.
    """
    hf_cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    volumes: dict[str, str] = {hf_cache: "/root/.cache/huggingface"}
    if extra:
        volumes.update(extra)
    return volumes


def merge_env(*env_dicts: dict[str, str] | None) -> dict[str, str]:
    """Merge multiple environment dicts (later values win)."""
    merged: dict[str, str] = {}
    for d in env_dicts:
        if d:
            merged.update(d)
    return merged


# ---------------------------------------------------------------------------
# InfiniBand detection flow
# ---------------------------------------------------------------------------

def detect_infiniband(
        hosts: list[str],
        head_host: str | None = None,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> dict[str, str]:
    """Run InfiniBand detection on *hosts* and return NCCL env vars.

    Detects IB on all hosts in parallel and uses the head node's (or
    the first host's) result to generate NCCL configuration.

    Args:
        hosts: Hosts to probe.
        head_host: Which host's IB config to use (defaults to ``hosts[0]``).
        ssh_kwargs: SSH connection parameters.
        dry_run: Log what would happen without executing.

    Returns:
        Dict of NCCL environment variables (empty if no IB found).
    """
    if not hosts:
        return {}

    target_host = head_host or hosts[0]
    kw = ssh_kwargs or {}

    logger.info("Detecting InfiniBand on %d host(s)...", len(hosts))
    ib_script = generate_ib_detect_script()
    ib_results = run_remote_scripts_parallel(
        hosts, ib_script, timeout=30, dry_run=dry_run, **kw,
    )

    nccl_env: dict[str, str] = {}
    for result in ib_results:
        if result.host == target_host and result.success:
            ib_info = parse_ib_detect_output(result.stdout)
            nccl_env = generate_nccl_env(ib_info)
            if nccl_env:
                logger.info("  InfiniBand detected on %s, NCCL configured", target_host)
            break

    if not nccl_env:
        logger.info("  No InfiniBand detected, using default networking")

    return nccl_env


def detect_infiniband_local(
        dry_run: bool = False,
) -> dict[str, str]:
    """Run InfiniBand detection locally and return NCCL env vars."""
    ib_script = generate_ib_detect_script()
    result = run_local_script(ib_script, dry_run=dry_run)
    if result.success:
        ib_info = parse_ib_detect_output(result.stdout)
        nccl_env = generate_nccl_env(ib_info)
        if nccl_env:
            logger.info("  InfiniBand detected locally, NCCL configured")
            return nccl_env
        logger.info("  No InfiniBand detected, using default networking")
    else:
        logger.warning(
            "  InfiniBand detection failed, continuing without: %s",
            result.stderr[:100],
        )
    return {}


# ---------------------------------------------------------------------------
# Container cleanup
# ---------------------------------------------------------------------------

def cleanup_containers(
        hosts: list[str],
        container_names: list[str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> None:
    """Stop and remove named containers on every host.

    Args:
        hosts: Target hosts.
        container_names: Container names to remove on each host.
        ssh_kwargs: SSH connection parameters.
        dry_run: Log without executing.
    """
    kw = ssh_kwargs or {}
    cmds = "; ".join(docker_stop_cmd(name) for name in container_names)
    for host in hosts:
        run_remote_command(host, cmds, timeout=30, dry_run=dry_run, **kw)


def cleanup_containers_local(
        container_names: list[str],
        dry_run: bool = False,
) -> None:
    """Stop and remove named containers locally."""
    cmds = "; ".join(docker_stop_cmd(name) for name in container_names)
    run_local_script("#!/bin/bash\n" + cmds, dry_run=dry_run)


# ---------------------------------------------------------------------------
# IP detection
# ---------------------------------------------------------------------------

def local_ip_for(target_host: str) -> str | None:
    """Return the local IP address on the interface that routes to *target_host*.

    Uses a UDP connect (no packets sent) to let the OS pick the right
    source address.  Falls back to ``socket.gethostname()`` on failure.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((target_host, 1))  # port is arbitrary; no traffic sent
            return s.getsockname()[0]
    except Exception:
        # Fall back to hostname if routing lookup fails (e.g. target
        # is not resolvable from the control machine itself).
        return socket.gethostname() or None


def detect_host_ip(
        host: str,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> str:
    """Detect the management IP of a remote host.

    Returns:
        The detected IPv4 address string.

    Raises:
        RuntimeError: If detection fails or result is not a valid IP.
    """
    from sparkrun.orchestration.scripts import generate_ip_detect_script

    kw = ssh_kwargs or {}
    ip_script = generate_ip_detect_script()
    result = run_remote_script(host, ip_script, timeout=15, dry_run=dry_run, **kw)

    if dry_run:
        return "<HEAD_IP>"

    if not result.success:
        raise RuntimeError("Failed to detect IP on %s: %s" % (host, result.stderr[:200]))

    ip = result.last_line.strip()
    if not is_valid_ip(ip):
        raise RuntimeError(
            "Could not determine IP from output on %s: %s" % (host, result.stdout[-200:])
        )
    return ip


# ---------------------------------------------------------------------------
# Container liveness
# ---------------------------------------------------------------------------

def is_container_running(
        host: str,
        container_name: str,
        ssh_kwargs: dict | None = None,
) -> bool:
    """Check whether a Docker container is running on a remote host.

    Args:
        host: Remote hostname.
        container_name: Docker container name.
        ssh_kwargs: SSH connection parameters.

    Returns:
        True if the container is currently running.
    """
    kw = ssh_kwargs or {}
    cmd = "docker inspect -f '{{.State.Running}}' %s 2>/dev/null" % container_name
    result = run_remote_command(host, cmd, timeout=10, **kw)
    return result.success and "true" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Port readiness polling
# ---------------------------------------------------------------------------

def wait_for_port(
        host: str,
        port: int,
        max_retries: int = 60,
        retry_interval: int = 2,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        container_name: str | None = None,
) -> bool:
    """Poll until a TCP port is listening on a remote host.

    Args:
        host: Remote hostname.
        port: Port to check.
        max_retries: Maximum number of retries.
        retry_interval: Seconds between retries.
        ssh_kwargs: SSH connection parameters.
        dry_run: Skip waiting in dry-run mode.
        container_name: If provided, verify the container is still
            running on each iteration.  Aborts early if the container
            has exited (e.g. crashed on startup).

    Returns:
        True if port became reachable, False if timed out or the
        container exited.
    """
    if dry_run:
        return True

    kw = ssh_kwargs or {}
    check_cmd = "nc -z localhost %d" % port
    for attempt in range(1, max_retries + 1):
        # Check container liveness before polling the port
        if container_name and attempt > 1:
            if not is_container_running(host, container_name, ssh_kwargs=kw):
                logger.error(
                    "  Container %s is no longer running on %s — aborting wait",
                    container_name, host,
                )
                return False

        result = run_remote_command(host, check_cmd, timeout=5, **kw)
        if result.success:
            logger.info("  Port %d ready after %ds", port, attempt * retry_interval)
            return True
        if attempt % 10 == 0:
            logger.info(
                "  Still waiting for port %d (%ds elapsed)...",
                port, attempt * retry_interval,
            )
        time.sleep(retry_interval)

    return False


# ---------------------------------------------------------------------------
# IP validation
# ---------------------------------------------------------------------------

def is_valid_ip(ip: str) -> bool:
    """Basic check if a string looks like an IPv4 address."""
    # TODO: either do regex or use ipaddress module
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

def run_local_script(script: str, dry_run: bool = False) -> RemoteResult:
    """Execute a script locally via subprocess.

    Args:
        script: Bash script content to execute.
        dry_run: If True, log the script but don't execute.

    Returns:
        RemoteResult with host set to ``"localhost"``.
    """
    if dry_run:
        script_lines = script.count('\n')
        logger.info("[dry-run] Would execute locally (%d lines, %d bytes)",
                    script_lines, len(script))
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


# ---------------------------------------------------------------------------
# Execution helpers (local-or-remote dispatch)
# ---------------------------------------------------------------------------

def run_script_on_host(
        host: str,
        script: str,
        ssh_kwargs: dict | None = None,
        timeout: int | None = None,
        dry_run: bool = False,
) -> RemoteResult:
    """Run a script on a host — dispatches to local or remote execution.

    If *host* is ``"localhost"``, ``"127.0.0.1"``, or empty, runs locally.
    Otherwise runs via SSH.
    """
    from sparkrun.hosts import is_local_host
    if is_local_host(host):
        return run_local_script(script, dry_run=dry_run)
    kw = ssh_kwargs or {}
    return run_remote_script(host, script, timeout=timeout, dry_run=dry_run, **kw)


def run_command_on_host(
        host: str,
        command: str,
        ssh_kwargs: dict | None = None,
        timeout: int | None = None,
        dry_run: bool = False,
) -> RemoteResult:
    """Run a command on a host — dispatches to local or remote execution."""
    from sparkrun.hosts import is_local_host
    if is_local_host(host):
        return run_local_script("#!/bin/bash\n" + command, dry_run=dry_run)
    kw = ssh_kwargs or {}
    return run_remote_command(host, command, timeout=timeout, dry_run=dry_run, **kw)
