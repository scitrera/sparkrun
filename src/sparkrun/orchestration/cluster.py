"""Multi-node Ray cluster launcher for sparkrun.

Orchestrates launching a Ray cluster across multiple DGX Spark hosts:

1. Clean up existing containers on all hosts.
2. Detect InfiniBand on all hosts (parallel).
3. Launch Ray head on first host.
4. Launch Ray workers on remaining hosts (parallel).
5. Execute serve command on head node.
"""

from __future__ import annotations

import logging
import time

from sparkrun.config import SparkrunConfig, DEFAULT_HF_CACHE_DIR
from sparkrun.orchestration.ssh import (
    run_remote_script,
    run_remote_command,
    run_remote_scripts_parallel,
)
from sparkrun.orchestration.infiniband import (
    generate_ib_detect_script,
    parse_ib_detect_output,
    generate_nccl_env,
)
from sparkrun.orchestration.scripts import (
    generate_ray_head_script,
    generate_ray_worker_script,
    generate_exec_serve_script,
)
from sparkrun.orchestration.docker import docker_stop_cmd, generate_container_name

logger = logging.getLogger(__name__)


def run_cluster(
    hosts: list[str],
    image: str,
    serve_command: str,
    cluster_id: str = "sparkrun0",
    ray_port: int = 46379,
    dashboard_port: int = 8265,
    dashboard: bool = False,
    env: dict[str, str] | None = None,
    runtime_cluster_env: dict[str, str] | None = None,
    cache_dir: str | None = None,
    config: SparkrunConfig | None = None,
    dry_run: bool = False,
    detached: bool = True,
    skip_ib_detect: bool = False,
) -> int:
    """Launch a multi-node Ray cluster and execute the serve command.

    The first host in *hosts* becomes the Ray head node; all remaining
    hosts become Ray workers.

    Args:
        hosts: List of hostnames/IPs (first = head, rest = workers).
        image: Container image to use.
        serve_command: The inference serve command to run on head.
        cluster_id: Identifier for container naming.
        ray_port: Ray GCS port.
        dashboard_port: Ray dashboard port.
        dashboard: Enable the Ray dashboard.
        env: Additional environment variables.
        runtime_cluster_env: Runtime-specific cluster env vars.
        cache_dir: HuggingFace cache directory.
        config: SparkrunConfig instance for SSH settings.
        dry_run: Show what would be done without executing.
        detached: Run serve command in background.
        skip_ib_detect: Skip InfiniBand detection.

    Returns:
        Exit code (0 = success).
    """
    if len(hosts) < 1:
        logger.error("At least one host is required")
        return 1

    head_host = hosts[0]
    worker_hosts = hosts[1:]
    head_container = generate_container_name(cluster_id, "head")
    worker_container = generate_container_name(cluster_id, "worker")
    hf_cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    volumes = {hf_cache: "/root/.cache/huggingface"}

    ssh_kwargs: dict = {}
    if config:
        ssh_kwargs["ssh_user"] = config.ssh_user
        ssh_kwargs["ssh_key"] = config.ssh_key
        ssh_kwargs["ssh_options"] = config.ssh_options

    all_env = dict(env or {})
    if runtime_cluster_env:
        all_env.update(runtime_cluster_env)

    _print_banner(hosts, image, cluster_id, ray_port, dashboard_port, serve_command, dry_run)

    # Step 1: Cleanup existing containers on all hosts
    t0 = time.monotonic()
    logger.info("Step 1/5: Cleaning up existing containers for cluster '%s'...", cluster_id)
    for host in hosts:
        cleanup_cmd = f"{docker_stop_cmd(head_container)}; {docker_stop_cmd(worker_container)}"
        run_remote_command(host, cleanup_cmd, timeout=30, dry_run=dry_run, **ssh_kwargs)
    logger.info("Step 1/5: Cleanup done (%.1fs)", time.monotonic() - t0)

    # Step 2: InfiniBand detection on all hosts (parallel)
    t0 = time.monotonic()
    nccl_env: dict[str, str] = {}
    if not skip_ib_detect:
        logger.info("Step 2/5: Detecting InfiniBand on all hosts...")
        ib_script = generate_ib_detect_script()
        ib_results = run_remote_scripts_parallel(
            hosts, ib_script, timeout=30, dry_run=dry_run, **ssh_kwargs
        )
        # Use the head node's IB config
        for result in ib_results:
            if result.host == head_host and result.success:
                ib_info = parse_ib_detect_output(result.stdout)
                nccl_env = generate_nccl_env(ib_info)
                if nccl_env:
                    logger.info("  InfiniBand detected on %s, NCCL configured", head_host)
                break
        if not nccl_env:
            logger.info("  No InfiniBand detected, using default networking")
        logger.info("Step 2/5: IB detection done (%.1fs)", time.monotonic() - t0)
    else:
        logger.info("Step 2/5: Skipping InfiniBand detection")

    # Step 3: Launch Ray head
    t0 = time.monotonic()
    logger.info("Step 3/5: Launching Ray head on %s...", head_host)
    head_script = generate_ray_head_script(
        image=image,
        container_name=head_container,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        dashboard=dashboard,
        env=all_env,
        volumes=volumes,
        nccl_env=nccl_env,
    )

    head_result = run_remote_script(head_host, head_script, timeout=120, dry_run=dry_run, **ssh_kwargs)
    if not head_result.success and not dry_run:
        logger.error("Failed to launch Ray head: %s", head_result.stderr)
        return 1

    # Extract head IP from script output (last line)
    head_ip = head_result.last_line if not dry_run else "<HEAD_IP>"
    if not dry_run and not _is_valid_ip(head_ip):
        logger.error(
            "Could not determine head IP from output: %s",
            head_result.stdout[-200:],
        )
        return 1

    logger.info("  Ray head launched. HEAD_IP=%s", head_ip)

    # Wait for Ray head to initialize
    if not dry_run:
        logger.info("  Waiting 5s for Ray head to initialize...")
        time.sleep(5)
    logger.info("Step 3/5: Ray head ready (%.1fs)", time.monotonic() - t0)

    # Step 4: Launch Ray workers (parallel)
    t0 = time.monotonic()
    if worker_hosts:
        logger.info(
            "Step 4/5: Launching %d Ray worker(s) on %s...",
            len(worker_hosts),
            ", ".join(worker_hosts),
        )
        worker_script = generate_ray_worker_script(
            image=image,
            container_name=worker_container,
            head_ip=head_ip,
            ray_port=ray_port,
            env=all_env,
            volumes=volumes,
            nccl_env=nccl_env,
        )

        worker_results = run_remote_scripts_parallel(
            worker_hosts, worker_script, timeout=120, dry_run=dry_run, **ssh_kwargs
        )

        failed_workers = [r for r in worker_results if not r.success and not dry_run]
        if failed_workers:
            for r in failed_workers:
                logger.warning(
                    "  Worker launch may have failed on %s: %s",
                    r.host,
                    r.stderr[:100],
                )

        if not dry_run:
            logger.info("  Waiting 3s for workers to connect to head at %s:%d...", head_ip, ray_port)
            time.sleep(3)
        logger.info("Step 4/5: Workers launched (%.1fs)", time.monotonic() - t0)
    else:
        logger.info("Step 4/5: No worker hosts, skipping")

    # Step 5: Execute serve command on head
    t0 = time.monotonic()
    logger.info("Step 5/5: Executing serve command on head node %s (container: %s)...", head_host, head_container)
    exec_script = generate_exec_serve_script(
        container_name=head_container,
        serve_command=serve_command,
        env=all_env,
        detached=detached,
    )

    _print_connection_info(
        head_host, head_ip, head_container,
        worker_hosts, worker_container, dashboard_port,
    )

    exec_result = run_remote_script(head_host, exec_script, timeout=60, dry_run=dry_run, **ssh_kwargs)
    logger.info("Step 5/5: Serve command dispatched (%.1fs)", time.monotonic() - t0)

    if dry_run:
        return 0
    return exec_result.returncode


def stop_cluster(
    hosts: list[str],
    cluster_id: str = "sparkrun0",
    config: SparkrunConfig | None = None,
    dry_run: bool = False,
) -> int:
    """Stop a running cluster by removing containers on all hosts.

    Args:
        hosts: List of hostnames/IPs in the cluster.
        cluster_id: Cluster identifier used when launching.
        config: SparkrunConfig instance for SSH settings.
        dry_run: Show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    head_container = generate_container_name(cluster_id, "head")
    worker_container = generate_container_name(cluster_id, "worker")

    ssh_kwargs: dict = {}
    if config:
        ssh_kwargs["ssh_user"] = config.ssh_user
        ssh_kwargs["ssh_key"] = config.ssh_key
        ssh_kwargs["ssh_options"] = config.ssh_options

    for host in hosts:
        cleanup_cmd = f"{docker_stop_cmd(head_container)}; {docker_stop_cmd(worker_container)}"
        run_remote_command(host, cleanup_cmd, dry_run=dry_run, **ssh_kwargs)

    logger.info("Cluster '%s' stopped on %d host(s)", cluster_id, len(hosts))
    return 0


def _is_valid_ip(ip: str) -> bool:
    """Basic check if a string looks like an IPv4 address."""
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def _print_banner(
    hosts: list[str],
    image: str,
    cluster_id: str,
    ray_port: int,
    dashboard_port: int,
    command: str,
    dry_run: bool,
) -> None:
    """Print launch information banner."""
    mode = "DRY-RUN" if dry_run else "LIVE"
    logger.info("=" * 60)
    logger.info("sparkrun Cluster Launcher")
    logger.info("=" * 60)
    logger.info("Cluster ID:     %s", cluster_id)
    logger.info("Image:          %s", image)
    logger.info("Head Node:      %s", hosts[0])
    logger.info(
        "Worker Nodes:   %s",
        ", ".join(hosts[1:]) if len(hosts) > 1 else "<none>",
    )
    logger.info("Ray Port:       %s", ray_port)
    logger.info("Dashboard Port: %s", dashboard_port)
    logger.info("Command:        %s", command[:80])
    logger.info("Mode:           %s", mode)
    logger.info("=" * 60)


def _print_connection_info(
    head_host: str,
    head_ip: str,
    head_container: str,
    worker_hosts: list[str],
    worker_container: str,
    dashboard_port: int,
) -> None:
    """Print connection/management info."""
    logger.info("=" * 60)
    logger.info(
        "To view logs:    ssh %s 'docker exec %s tail -f /tmp/sparkrun_serve.log'",
        head_host,
        head_container,
    )
    logger.info(
        "To stop cluster: sparkrun stop --hosts %s --cluster-id ...",
        ",".join([head_host] + worker_hosts),
    )
    logger.info("Dashboard:       http://%s:%d", head_ip, dashboard_port)
    logger.info("=" * 60)
