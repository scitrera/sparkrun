"""Job and cluster metadata storage.

Persists cluster_id → recipe mapping in ``~/.cache/sparkrun/jobs/`` so
``cluster status`` and other commands can display recipe info for
running clusters.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)


def generate_cluster_id(recipe: "Recipe", hosts: list[str]) -> str:
    """Deterministic cluster identifier from recipe and host set.

    Takes the full Recipe and host list so the hash inputs can be
    expanded later (e.g. adding port, tensor_parallel) without
    changing the function signature.

    Currently hashes: runtime + model + sorted hosts.
    """
    parts = [recipe.runtime, recipe.model] + sorted(hosts)
    key = "\0".join(parts)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return "sparkrun_%s" % digest


def save_job_metadata(
    cluster_id: str,
    recipe: "Recipe",
    hosts: list[str],
    overrides: dict | None = None,
    cache_dir: str | None = None,
    ib_ip_map: dict[str, str] | None = None,
    mgmt_ip_map: dict[str, str] | None = None,
    recipe_ref: str | None = None,
) -> None:
    """Persist job metadata so ``cluster status`` can display recipe info.

    Writes a small YAML file to ``{cache_dir}/jobs/{hash}.yaml`` where
    *hash* is the 12-char hex portion of *cluster_id*.
    """
    if cache_dir is None:
        from sparkrun.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    jobs_dir = Path(cache_dir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    tp = None
    if overrides:
        tp = overrides.get("tensor_parallel")
    if tp is None and recipe.defaults:
        tp = recipe.defaults.get("tensor_parallel")

    meta = {
        "cluster_id": cluster_id,
        "recipe": recipe.name,
        "model": recipe.model,
        "runtime": recipe.runtime,
        "hosts": hosts,
    }
    if recipe_ref:
        meta["recipe_ref"] = recipe_ref
    if tp is not None:
        meta["tensor_parallel"] = int(tp)
    if ib_ip_map:
        meta["ib_ip_map"] = ib_ip_map
    if mgmt_ip_map:
        meta["mgmt_ip_map"] = mgmt_ip_map

    meta_path = jobs_dir / f"{digest}.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    logger.debug("Saved job metadata to %s", meta_path)


def remove_job_metadata(cluster_id: str, cache_dir: str | None = None) -> None:
    """Delete the cached job metadata file for a cluster_id.

    No-op if the file does not exist.
    """
    if cache_dir is None:
        from sparkrun.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    meta_path.unlink(missing_ok=True)
    logger.debug("Removed job metadata %s", meta_path)


def load_job_metadata(cluster_id: str, cache_dir: str | None = None) -> dict | None:
    """Load job metadata for a cluster_id.  Returns ``None`` if not found."""
    if cache_dir is None:
        from sparkrun.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    if not meta_path.exists():
        return None
    try:
        from sparkrun.utils import load_yaml
        data = load_yaml(meta_path)
        return data or None
    except Exception:
        logger.debug("Failed to load job metadata for %s", cluster_id, exc_info=True)
        return None
