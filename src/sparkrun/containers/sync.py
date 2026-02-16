"""Container image distribution across cluster nodes."""

from __future__ import annotations

import logging

from sparkrun.orchestration.ssh import run_remote_scripts_parallel
from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)


def sync_image_to_hosts(
    image: str,
    hosts: list[str],
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Ensure a container image is available on all hosts.

    Checks each host in parallel and pulls the image where missing.

    Args:
        image: Container image reference.
        hosts: List of remote hostnames or IPs.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        dry_run: If True, show what would be done without executing.

    Returns:
        List of hostnames where the image sync failed.
    """
    script = read_script("image_sync.sh").format(image=image)

    results = run_remote_scripts_parallel(
        hosts,
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )

    failed = [r.host for r in results if not r.success]
    if failed:
        logger.warning("Image sync failed on hosts: %s", failed)
    else:
        logger.info("Image available on all %d hosts", len(hosts))

    return failed
