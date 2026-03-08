"""Model load/unload via sparkrun subprocess delegation.

Reuses existing ``sparkrun run`` and ``sparkrun stop`` commands
via subprocess to load and unload models, preserving all existing
orchestration logic (distribution, IB detection, etc.).
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def load_model(
    recipe_name: str,
    cluster: str | None = None,
    hosts: str | None = None,
    hosts_file: str | None = None,
    port: int | None = None,
    overrides: dict | None = None,
    dry_run: bool = False,
) -> bool:
    """Load a model via ``sparkrun run``.

    Args:
        recipe_name: Recipe name or path.
        cluster: Named cluster to use.
        hosts: Comma-separated host list.
        hosts_file: Path to hosts file.
        port: Override serve port.
        overrides: Additional key=value overrides.
        dry_run: Show what would be done.

    Returns:
        True if the model was launched successfully.
    """
    sparkrun = _find_sparkrun()
    if not sparkrun:
        return False

    cmd = [sparkrun, "run", recipe_name, "--no-follow"]

    if cluster:
        cmd.extend(["--cluster", cluster])
    if hosts:
        cmd.extend(["--hosts", hosts])
    if hosts_file:
        cmd.extend(["--hosts-file", hosts_file])
    if port is not None:
        cmd.extend(["--port", str(port)])
    if overrides:
        for key, value in overrides.items():
            cmd.extend(["-o", "%s=%s" % (key, value)])
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Loading model: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to load model: %s", result.stderr[:500] if result.stderr else "unknown error")
        return result.returncode == 0
    except FileNotFoundError:
        logger.error("sparkrun command not found")
        return False


def unload_model(
    recipe_name: str,
    cluster: str | None = None,
    hosts: str | None = None,
    hosts_file: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Unload a model via ``sparkrun stop``.

    Args:
        recipe_name: Recipe name or path.
        cluster: Named cluster to use.
        hosts: Comma-separated host list.
        hosts_file: Path to hosts file.
        dry_run: Show what would be done.

    Returns:
        True if the model was stopped successfully.
    """
    sparkrun = _find_sparkrun()
    if not sparkrun:
        return False

    cmd = [sparkrun, "stop", recipe_name]

    if cluster:
        cmd.extend(["--cluster", cluster])
    if hosts:
        cmd.extend(["--hosts", hosts])
    if hosts_file:
        cmd.extend(["--hosts-file", hosts_file])
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Unloading model: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to unload model: %s", result.stderr[:500] if result.stderr else "unknown error")
        return result.returncode == 0
    except FileNotFoundError:
        logger.error("sparkrun command not found")
        return False


def _find_sparkrun() -> str | None:
    """Locate the sparkrun binary on PATH."""
    path = shutil.which("sparkrun")
    if not path:
        logger.error("sparkrun command not found on PATH")
    return path
