"""Model distribution across cluster nodes."""

from __future__ import annotations

import logging

from sparkrun.config import DEFAULT_HF_CACHE_DIR
from sparkrun.orchestration.ssh import run_remote_scripts_parallel
from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)


def sync_model_to_hosts(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Download a model on all hosts in parallel.

    Uses ``huggingface-cli`` on each remote host to download the model
    if it is not already cached.

    Args:
        model_id: HuggingFace model identifier.
        hosts: List of remote hostnames or IPs.
        cache_dir: Override for the HuggingFace cache directory.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        dry_run: If True, show what would be done without executing.

    Returns:
        List of hostnames where the sync failed.
    """
    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)

    script = read_script("model_sync.sh").format(model_id=model_id, cache=cache)

    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )

    failed = [r.host for r in results if not r.success]
    if failed:
        logger.warning("Model sync failed on hosts: %s", failed)
    else:
        logger.info("Model synced to all %d hosts", len(hosts))

    return failed
