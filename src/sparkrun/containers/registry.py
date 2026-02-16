"""Container image registry operations."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def pull_image(image: str, dry_run: bool = False) -> int:
    """Pull a container image from a registry.

    Args:
        image: Image reference to pull (e.g. ``"nvcr.io/nvidia/vllm:latest"``).
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    if dry_run:
        logger.info("[dry-run] Would pull image: %s", image)
        return 0

    logger.info("Pulling image: %s...", image)
    result = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to pull image %s: %s", image, result.stderr[:200])
    return result.returncode


def image_exists_locally(image: str) -> bool:
    """Check if a container image exists locally.

    Args:
        image: Image reference to check.

    Returns:
        True if the image exists in the local Docker image store.
    """
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def ensure_image(image: str, dry_run: bool = False) -> int:
    """Ensure an image exists locally, pulling if needed.

    Args:
        image: Image reference.
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    if image_exists_locally(image):
        logger.info("Image already available: %s", image)
        return 0
    return pull_image(image, dry_run=dry_run)
