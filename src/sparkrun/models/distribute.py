"""Model distribution via local-to-remote transfer.

Instead of having every host download from HuggingFace Hub independently,
these functions download once (locally or on the head node) and then
rsync the cache directory to targets.
"""

from __future__ import annotations

import logging

from sparkrun.config import DEFAULT_HF_CACHE_DIR
from sparkrun.models.download import download_model
from sparkrun.orchestration.ssh import (
    build_ssh_opts_string,
    run_remote_script,
    run_rsync_parallel,
)
from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)


def _model_cache_path(model_id: str, cache_dir: str) -> str:
    """Compute the HF cache path for a model.

    Mirrors the logic in :func:`sparkrun.models.download.is_model_cached`.
    """
    safe_name = model_id.replace("/", "--")
    return f"{cache_dir}/hub/models--{safe_name}"


def distribute_model_from_local(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    token: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    transfer_hosts: list[str] | None = None,
) -> list[str]:
    """Download a model locally then rsync it to all hosts.

    1. Download the model to the local HF cache via :func:`download_model`.
    2. For each host in parallel, rsync the model cache directory.
       Because rsync is incremental, hosts that already have the model
       will complete almost instantly.

    Args:
        model_id: HuggingFace model identifier.
        hosts: Target hostnames or IPs (used for identification/reporting).
        cache_dir: Override for the HuggingFace cache directory.
        token: Optional HuggingFace API token for gated models.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-host transfer timeout in seconds.
        dry_run: If True, show what would be done without executing.
        transfer_hosts: Optional IB/fast-network IPs to use for the actual
            data transfer.  Must be same length as *hosts*.  When provided,
            ``transfer_hosts[i]`` is used for SSH connections while
            ``hosts[i]`` is used for identification and error reporting.
            Falls back to *hosts* when ``None``.

    Returns:
        List of hostnames (from *hosts*) where distribution failed
        (empty = full success).
    """
    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    logger.info("Distributing model '%s' from local to %d host(s)", model_id, len(hosts))

    # Step 1: download model locally
    rc = download_model(model_id, cache_dir=cache, token=token, dry_run=dry_run)
    if rc != 0:
        logger.error("Failed to download model '%s' locally — aborting distribution", model_id)
        return list(hosts)

    if not hosts:
        return []

    xfer = transfer_hosts or hosts

    # Step 2: rsync model cache to all hosts in parallel
    model_path = _model_cache_path(model_id, cache)
    results = run_rsync_parallel(
        model_path, xfer, model_path,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        timeout=timeout, dry_run=dry_run,
    )

    # Map transfer IPs back to management hosts for failure reporting
    xfer_to_host = dict(zip(xfer, hosts))
    failed = [xfer_to_host.get(r.host, r.host) for r in results if not r.success]
    if failed:
        logger.warning("Model distribution failed on hosts: %s", failed)
    else:
        logger.info("Model '%s' distributed to all %d host(s)", model_id, len(hosts))

    return failed


def distribute_model_from_head(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
    worker_transfer_hosts: list[str] | None = None,
) -> list[str]:
    """Download a model on the head node then distribute to remaining hosts.

    1. Download on ``hosts[0]`` using the existing ``model_sync.sh``.
    2. If there is only one host, done.
    3. Run ``model_distribute.sh`` on ``hosts[0]`` to rsync to ``hosts[1:]``.

    Args:
        model_id: HuggingFace model identifier.
        hosts: Cluster hostnames (``hosts[0]`` is the head).
        cache_dir: Override for the HuggingFace cache directory.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-operation timeout in seconds.
        dry_run: If True, show what would be done without executing.
        worker_transfer_hosts: Optional IB/fast-network IPs for workers
            (``hosts[1:]``).  Used as targets in the distribution script
            running on the head.  Falls back to ``hosts[1:]`` when ``None``.

    Returns:
        List of hostnames where distribution failed (empty = full success).
    """
    if not hosts:
        return []

    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    head = hosts[0]
    logger.info("Distributing model '%s' from head (%s) to %d host(s)",
                model_id, head, len(hosts))

    # Step 1: download model on head
    dl_script = read_script("model_sync.sh").format(model_id=model_id, cache=cache)
    dl_result = run_remote_script(
        head, dl_script,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        timeout=timeout, dry_run=dry_run,
    )
    if not dl_result.success:
        logger.error("Failed to download model on head %s", head)
        return list(hosts)

    # Step 2: if single host, we're done
    if len(hosts) == 1:
        logger.info("Single host — model download complete")
        return []

    # Step 3: distribute from head to remaining hosts
    targets = worker_transfer_hosts or hosts[1:]
    model_path = _model_cache_path(model_id, cache)
    ssh_opts = build_ssh_opts_string(
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
    )
    dist_script = read_script("model_distribute.sh").format(
        model_path=model_path,
        targets=" ".join(targets),
        ssh_opts=ssh_opts,
        ssh_user=ssh_user or "",
    )
    dist_result = run_remote_script(
        head, dist_script,
        ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options,
        timeout=timeout, dry_run=dry_run,
    )

    if dist_result.success:
        logger.info("Model '%s' distributed from head to all targets", model_id)
        return []

    # Report using management hostnames
    logger.warning("Model distribution from head failed (rc=%d)", dist_result.returncode)
    return list(hosts[1:])
