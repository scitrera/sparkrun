"""Resource distribution: IB detection, container image and model syncing."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from sparkrun.hosts import is_local_host

if TYPE_CHECKING:
    from sparkrun.config import SparkrunConfig

logger = logging.getLogger(__name__)


def distribute_resources(
        image: str,
        model: str,
        host_list: list[str],
        cache_dir: str,
        config: SparkrunConfig,
        dry_run: bool,
        skip_ib: bool,
        model_revision: str | None = None,
        recipe_name: str = "",
) -> tuple[dict[str, str] | None, dict[str, str], dict[str, str]]:
    """Detect IB, distribute container image and model to target hosts.

    Performs InfiniBand detection (for both NCCL env and IB transfer IPs),
    then distributes the container image and model from local to all
    remote hosts using the fast IB network when available.

    For localhost targets, only ensures the image/model exist locally.

    Args:
        image: Container image reference.
        model: HuggingFace model identifier (may be empty).
        host_list: Target hostnames/IPs.
        cache_dir: HuggingFace cache directory.
        config: SparkrunConfig instance.
        dry_run: Show what would be done without executing.
        skip_ib: Skip InfiniBand detection.
        model_revision: Optional HuggingFace model revision to pin.
        recipe_name: Recipe name for pending-op lock display.

    Returns:
        Tuple of (nccl_env, ib_ip_map, mgmt_ip_map).  ``nccl_env`` is
        ``None`` when IB detection was skipped or not applicable.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts
    from sparkrun.containers.distribute import distribute_image_from_local
    from sparkrun.containers.registry import ensure_image
    from sparkrun.models.distribute import distribute_model_from_local
    from sparkrun.models.download import download_model
    from sparkrun.pending_ops import pending_op

    # Common kwargs for pending-op lock files
    _pop_kw = dict(
        recipe=recipe_name,
        model=model, image=image,
        hosts=host_list, cache_dir=str(config.cache_dir),
    )
    # Derive a cluster_id-ish key for the lock files.  The real cluster_id
    # is generated earlier in run(); we receive the image+model+hosts here
    # so we hash the same inputs to keep the lock name stable.
    _lock_key = hashlib.sha256(
        f"{image}|{model}|{','.join(host_list)}".encode()
    ).hexdigest()[:12]
    _lock_id = f"sparkrun_{_lock_key}"

    if is_local := (len(host_list) <= 1 and is_local_host(host_list[0])):
        # Local-only: just ensure image and model exist, no SSH needed
        with pending_op(_lock_id, "image_pull", **_pop_kw):
            logger.info("Ensuring container image is available locally...")
            ensure_image(image, dry_run=dry_run)
        if model:
            with pending_op(_lock_id, "model_download", **_pop_kw):
                logger.info("Ensuring model %s is available locally...", model)
                download_model(model, cache_dir=cache_dir, revision=model_revision, dry_run=dry_run)
        return None, {}, {}  # let runtime handle its own local IB detection

    ssh_kwargs = build_ssh_kwargs(config)
    nccl_env: dict[str, str] = {}
    ib_ip_map: dict[str, str] = {}
    mgmt_ip_map: dict[str, str] = {}
    transfer_hosts: list[str] | None = None

    # Step 1: Detect InfiniBand for NCCL env + transfer routing
    if not skip_ib:
        ib_result = detect_ib_for_hosts(
            host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        nccl_env = ib_result.nccl_env
        ib_ip_map = ib_result.ib_ip_map
        mgmt_ip_map = ib_result.mgmt_ip_map
        if ib_result.ib_ip_map:
            transfer_hosts = [
                ib_result.ib_ip_map.get(h, h) for h in host_list
            ]
            logger.info(
                "Using IB network for transfers (%d/%d hosts)",
                len(ib_result.ib_ip_map), len(host_list),
            )

    # Step 2: Distribute container image
    with pending_op(_lock_id, "image_distribute", **_pop_kw):
        img_failed = distribute_image_from_local(
            image, host_list,
            transfer_hosts=transfer_hosts,
            dry_run=dry_run, **ssh_kwargs,
        )
    if img_failed:
        logger.warning("Image distribution failed on: %s", ", ".join(img_failed))

    # Step 3: Distribute model
    if model:
        with pending_op(_lock_id, "model_download", **_pop_kw):
            mdl_failed = distribute_model_from_local(
                model, host_list,
                cache_dir=cache_dir,
                revision=model_revision,
                transfer_hosts=transfer_hosts,
                dry_run=dry_run, **ssh_kwargs,
            )
        if mdl_failed:
            logger.warning("Model distribution failed on: %s", ", ".join(mdl_failed))

    logger.info("Distribution complete.")
    return nccl_env, ib_ip_map, mgmt_ip_map
