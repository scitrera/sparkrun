"""Named cluster management for SparkRun."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Name validation pattern: start with alphanumeric, contain alphanumeric/underscore/hyphen
CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class ClusterError(Exception):
    """Raised when cluster operations fail."""

    pass


@dataclass
class ClusterDefinition:
    """Definition of a named cluster."""

    name: str
    hosts: list[str]
    description: str = ""


class ClusterManager:
    """Manages named cluster definitions stored as YAML files."""

    def __init__(self, config_root: Path) -> None:
        """Initialize cluster manager.

        Args:
            config_root: Base configuration directory (e.g. ~/.config/sparkrun/).
                Cluster files will be stored in config_root/clusters/.
        """
        self.config_root = Path(config_root)
        self.clusters_dir = self.config_root / "clusters"
        self.default_file = self.clusters_dir / ".default"

        # Ensure clusters directory exists
        self.clusters_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ClusterManager initialized with clusters_dir: %s", self.clusters_dir)

    def _validate_name(self, name: str) -> None:
        """Validate cluster name against allowed pattern.

        Args:
            name: Cluster name to validate

        Raises:
            ClusterError: If name is invalid
        """
        if not CLUSTER_NAME_PATTERN.match(name):
            raise ClusterError(
                f"Invalid cluster name '{name}': must start with alphanumeric character "
                "and contain only alphanumeric, underscore, or hyphen characters"
            )

    def _cluster_path(self, name: str) -> Path:
        """Get path to cluster YAML file."""
        return self.clusters_dir / f"{name}.yaml"

    def create(self, name: str, hosts: list[str], description: str = "") -> None:
        """Create a new named cluster.

        Args:
            name: Cluster name
            hosts: List of host addresses
            description: Optional cluster description

        Raises:
            ClusterError: If cluster already exists or name is invalid
        """
        self._validate_name(name)

        cluster_path = self._cluster_path(name)
        if cluster_path.exists():
            raise ClusterError(f"Cluster '{name}' already exists")

        cluster_def = ClusterDefinition(name=name, hosts=hosts, description=description)
        self._write_cluster(cluster_def)
        logger.info("Created cluster '%s' with %d hosts", name, len(hosts))

    def get(self, name: str) -> ClusterDefinition:
        """Load cluster definition by name.

        Args:
            name: Cluster name

        Returns:
            ClusterDefinition for the requested cluster

        Raises:
            ClusterError: If cluster not found
        """
        cluster_path = self._cluster_path(name)
        if not cluster_path.exists():
            raise ClusterError(f"Cluster '{name}' not found")

        return self._read_cluster(cluster_path)

    def update(self, name: str, hosts: list[str] | None = None, description: str | None = None) -> None:
        """Update existing cluster definition.

        Args:
            name: Cluster name
            hosts: New host list (if provided)
            description: New description (if provided)

        Raises:
            ClusterError: If cluster does not exist
        """
        # Load existing cluster
        cluster_def = self.get(name)

        # Update provided fields
        if hosts is not None:
            cluster_def.hosts = hosts
            logger.debug("Updated hosts for cluster '%s'", name)

        if description is not None:
            cluster_def.description = description
            logger.debug("Updated description for cluster '%s'", name)

        # Write back
        self._write_cluster(cluster_def)
        logger.info("Updated cluster '%s'", name)

    def list_clusters(self) -> list[ClusterDefinition]:
        """List all defined clusters.

        Returns:
            List of ClusterDefinition objects sorted by name
        """
        clusters = []
        for yaml_file in self.clusters_dir.glob("*.yaml"):
            try:
                cluster_def = self._read_cluster(yaml_file)
                clusters.append(cluster_def)
            except Exception as e:
                logger.warning("Failed to load cluster from %s: %s", yaml_file, e)

        clusters.sort(key=lambda c: c.name)
        logger.debug("Listed %d clusters", len(clusters))
        return clusters

    def delete(self, name: str) -> None:
        """Delete a cluster definition.

        Args:
            name: Cluster name

        Raises:
            ClusterError: If cluster not found
        """
        cluster_path = self._cluster_path(name)
        if not cluster_path.exists():
            raise ClusterError(f"Cluster '{name}' not found")

        cluster_path.unlink()
        logger.info("Deleted cluster '%s'", name)

        # Clear default if it pointed to this cluster
        default_name = self.get_default()
        if default_name == name:
            self.unset_default()
            logger.debug("Cleared default pointer as it referenced deleted cluster '%s'", name)

    def set_default(self, name: str) -> None:
        """Set the default cluster.

        Args:
            name: Cluster name

        Raises:
            ClusterError: If cluster does not exist
        """
        # Verify cluster exists
        self.get(name)

        self.default_file.write_text(name)
        logger.info("Set default cluster to '%s'", name)

    def unset_default(self) -> None:
        """Clear the default cluster marker.

        Does not raise error if default is not set.
        """
        if self.default_file.exists():
            self.default_file.unlink()
            logger.debug("Unset default cluster")

    def get_default(self) -> str | None:
        """Get the default cluster name.

        Returns:
            Default cluster name if set and cluster exists, None otherwise
        """
        if not self.default_file.exists():
            return None

        default_name = self.default_file.read_text().strip()
        if not default_name:
            return None

        # Verify cluster still exists
        cluster_path = self._cluster_path(default_name)
        if not cluster_path.exists():
            logger.warning("Default cluster '%s' no longer exists, clearing default", default_name)
            self.unset_default()
            return None

        return default_name

    def _write_cluster(self, cluster_def: ClusterDefinition) -> None:
        """Write cluster definition to YAML file."""
        cluster_path = self._cluster_path(cluster_def.name)

        data: dict[str, Any] = {
            "name": cluster_def.name,
            "hosts": cluster_def.hosts,
            "description": cluster_def.description,
        }

        with cluster_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.debug("Wrote cluster definition to %s", cluster_path)

    def _read_cluster(self, cluster_path: Path) -> ClusterDefinition:
        """Read cluster definition from YAML file."""
        with cluster_path.open("r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ClusterError(f"Invalid cluster file format: {cluster_path}")

        return ClusterDefinition(
            name=data.get("name", ""),
            hosts=data.get("hosts", []),
            description=data.get("description", ""),
        )
