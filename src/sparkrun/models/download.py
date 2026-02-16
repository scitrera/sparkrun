"""HuggingFace model download utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from sparkrun.config import DEFAULT_HF_CACHE_DIR

logger = logging.getLogger(__name__)


def is_model_cached(model_id: str, cache_dir: str | None = None) -> bool:
    """Check if a model is already cached locally.

    Inspects the HuggingFace cache directory structure to determine
    whether the model has been previously downloaded.

    Args:
        model_id: HuggingFace model identifier (e.g. ``"meta-llama/Llama-3-8B"``).
        cache_dir: Override for the HuggingFace cache directory.

    Returns:
        True if the model cache directory exists and is non-empty.
    """
    cache = Path(cache_dir or str(DEFAULT_HF_CACHE_DIR))
    # HF cache structure: hub/models--org--name/
    safe_name = model_id.replace("/", "--")
    model_cache = cache / "hub" / f"models--{safe_name}"
    return model_cache.exists() and any(model_cache.iterdir())


def download_model(
    model_id: str,
    cache_dir: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> int:
    """Download a model from HuggingFace Hub.

    Uses the ``huggingface_hub`` Python API for robust downloading.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Override for the HuggingFace cache directory.
        token: Optional HuggingFace API token for gated models.
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)

    if dry_run:
        logger.info("[dry-run] Would download model: %s to %s", model_id, cache)
        return 0

    if is_model_cached(model_id, cache):
        logger.info("Model %s already cached at %s", model_id, cache)
        return 0

    logger.info("Downloading model: %s...", model_id)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, cache_dir=cache, token=token)
        logger.info("Model downloaded successfully: %s", model_id)
        return 0
    except Exception as e:
        logger.error("Failed to download model %s: %s", model_id, e)
        return 1
