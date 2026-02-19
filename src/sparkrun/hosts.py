"""Host resolution with priority chain.

Resolves hosts from CLI args, files, cluster manager, or config defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sparkrun.cluster_manager import ClusterError, ClusterManager

logger = logging.getLogger(__name__)


def is_local_host(host: str) -> bool:
    """Check if a host string refers to the local machine."""
    return host in ("localhost", "127.0.0.1", "")


class HostResolutionError(Exception):
    """Error during host resolution."""

    pass


def parse_hosts_file(path: str | Path) -> list[str]:
    """Parse hosts file with one host per line.

    Comments (#) and blank lines are ignored.

    Args:
        path: Path to hosts file

    Returns:
        List of host strings

    Raises:
        HostResolutionError: If file not found
    """
    file_path = Path(path)
    if not file_path.exists():
        raise HostResolutionError("Hosts file not found: %s" % file_path)

    hosts = []
    with file_path.open("r") as f:
        for line in f:
            # Strip comments
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if line:
                hosts.append(line)

    logger.debug("Parsed %d hosts from file: %s", len(hosts), file_path)
    return hosts


def resolve_hosts(
    hosts: str | None = None,
    hosts_file: str | None = None,
    cluster_name: str | None = None,
    cluster_manager: ClusterManager | None = None,
    config_default_hosts: list[str] | None = None,
) -> list[str]:
    """Resolve hosts using priority chain.

    Priority:
    1. hosts (comma-separated CLI arg)
    2. hosts_file (path to file with one host per line)
    3. cluster_name (named cluster via ClusterManager)
    4. Default cluster from ClusterManager
    5. config_default_hosts (legacy fallback from config.yaml)
    6. Empty list (caller decides whether to error)

    Args:
        hosts: Comma-separated host list
        hosts_file: Path to hosts file
        cluster_name: Named cluster lookup
        cluster_manager: ClusterManager instance
        config_default_hosts: Legacy config default hosts

    Returns:
        List of resolved host strings
    """
    # Priority 1: CLI hosts arg
    if hosts:
        resolved = [h.strip() for h in hosts.split(",") if h.strip()]
        logger.debug("Resolved %d hosts from CLI arg", len(resolved))
        return resolved

    # Priority 2: hosts_file
    if hosts_file:
        return parse_hosts_file(hosts_file)

    # Priority 3: named cluster
    if cluster_name and cluster_manager:
        try:
            cluster = cluster_manager.get(cluster_name)
            resolved = cluster.hosts
            logger.debug(
                "Resolved %d hosts from cluster '%s'", len(resolved), cluster_name
            )
            return resolved
        except ClusterError as e:
            logger.warning("Failed to resolve cluster '%s': %s", cluster_name, e)

    # Priority 4: default cluster
    if cluster_manager:
        try:
            default_name = cluster_manager.get_default()
            if default_name:
                cluster = cluster_manager.get(default_name)
                resolved = cluster.hosts
                logger.debug(
                    "Resolved %d hosts from default cluster '%s'",
                    len(resolved),
                    cluster.name,
                )
                return resolved
        except ClusterError as e:
            logger.debug("No default cluster available: %s", e)

    # Priority 5: config default hosts
    if config_default_hosts:
        logger.debug("Resolved %d hosts from config defaults", len(config_default_hosts))
        return config_default_hosts

    # Priority 6: empty list
    logger.debug("No hosts resolved from any source")
    return []
