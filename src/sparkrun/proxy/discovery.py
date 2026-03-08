"""Endpoint discovery from job metadata and live health checks.

Scans ``~/.cache/sparkrun/jobs/*.yaml`` metadata files, extracts
endpoint information, deduplicates by host:port (since multiple
stale metadata entries may point to the same running server), and
health-checks each unique endpoint to determine what's actually
reachable.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = 3  # seconds


@dataclass
class DiscoveredEndpoint:
    """A discovered inference endpoint from job metadata."""

    cluster_id: str
    model: str
    served_model_name: str | None
    runtime: str
    host: str
    port: int
    healthy: bool
    actual_models: list[str] = field(default_factory=list)
    recipe_name: str = ""
    tensor_parallel: int = 1


def discover_endpoints(
    host_filter: list[str] | None = None,
    cache_dir: str | None = None,
    check_health: bool = True,
) -> list[DiscoveredEndpoint]:
    """Discover running inference endpoints from job metadata.

    Pipeline:
    1. Glob ``~/.cache/sparkrun/jobs/*.yaml`` and load each.
    2. Extract head host, port, model, runtime from metadata.
    3. Optionally filter by host list.
    4. **Deduplicate by host:port** — multiple stale metadata entries
       may reference the same server; keep the most recent file.
    5. Parallel health checks via GET ``/v1/models``.
    6. Return only healthy endpoints (unhealthy/stale entries filtered out).

    Args:
        host_filter: Only include endpoints on these hosts.
        cache_dir: Override cache directory (default: ~/.cache/sparkrun).
        check_health: Whether to perform HTTP health checks.

    Returns:
        List of discovered endpoints (healthy only when check_health=True).
    """
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    jobs_dir = Path(cache_dir) / "jobs"
    if not jobs_dir.is_dir():
        return []

    from sparkrun.utils import load_yaml

    # Collect all candidates, keyed by host:port.
    # Later entries (sorted by filename) overwrite earlier ones,
    # so the most recent metadata wins for each host:port.
    candidates: dict[str, DiscoveredEndpoint] = {}

    for meta_path in sorted(jobs_dir.glob("*.yaml")):
        try:
            meta = load_yaml(meta_path)
        except Exception:
            logger.debug("Failed to load job metadata: %s", meta_path, exc_info=True)
            continue

        if not meta:
            continue

        hosts = meta.get("hosts", [])
        if not hosts:
            continue

        head_host = hosts[0]

        # Apply host filter
        if host_filter and head_host not in host_filter:
            continue

        port = int(meta.get("port", 8000))
        model = meta.get("model", "")
        runtime = meta.get("runtime", "")
        cluster_id = meta.get("cluster_id", "")
        recipe_name = meta.get("recipe", meta.get("recipe_ref", ""))
        served_name = meta.get("served_model_name")
        tp = int(meta.get("tensor_parallel", 1))

        # Deduplicate: one entry per host:port (last write wins)
        key = "%s:%d" % (head_host, port)
        candidates[key] = DiscoveredEndpoint(
            cluster_id=cluster_id,
            model=model,
            served_model_name=served_name,
            runtime=runtime,
            host=head_host,
            port=port,
            healthy=False,
            recipe_name=recipe_name,
            tensor_parallel=tp,
        )

    endpoints = list(candidates.values())

    if check_health and endpoints:
        _check_health_parallel(endpoints)
        # Only return healthy endpoints — stale metadata for dead
        # servers is not useful to callers.
        endpoints = [ep for ep in endpoints if ep.healthy]
        # Deduplicate endpoints serving identical models on the same
        # port but reachable via different network interfaces (e.g.
        # management IP vs ConnectX-7 IP).  Keep the first one found
        # (management IP is typically listed first in metadata).
        endpoints = _deduplicate_by_identity(endpoints)

    return endpoints


def _check_health_parallel(endpoints: list[DiscoveredEndpoint]) -> None:
    """Run health checks in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=min(len(endpoints), 8)) as pool:
        futures = {
            pool.submit(_check_single_health, ep): ep
            for ep in endpoints
        }
        for future in as_completed(futures):
            ep = futures[future]
            try:
                healthy, models = future.result()
                ep.healthy = healthy
                ep.actual_models = models
            except Exception:
                logger.debug(
                    "Health check failed for %s:%d", ep.host, ep.port,
                    exc_info=True,
                )


def _deduplicate_by_identity(endpoints: list[DiscoveredEndpoint]) -> list[DiscoveredEndpoint]:
    """Collapse endpoints that serve the same models on the same port.

    When a DGX Spark is reachable via both management and ConnectX-7
    IPs, two metadata entries may point to the same running server.
    After health checks confirm they serve the same model set, keep
    only the first occurrence (management IP is typically earlier in
    the sorted metadata files).
    """
    seen: dict[tuple, DiscoveredEndpoint] = {}
    for ep in endpoints:
        identity = (frozenset(ep.actual_models), ep.port) if ep.actual_models else (ep.model, ep.port)
        if identity not in seen:
            seen[identity] = ep
        else:
            logger.debug(
                "Dedup: %s:%d is same server as %s:%d (same models on port %d)",
                ep.host, ep.port, seen[identity].host, seen[identity].port, ep.port,
            )
    return list(seen.values())


def _check_single_health(ep: DiscoveredEndpoint) -> tuple[bool, list[str]]:
    """Check a single endpoint's health via GET /v1/models.

    Returns:
        Tuple of (healthy, list_of_model_ids).
    """
    url = "http://%s:%d/v1/models" % (ep.host, ep.port)
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_HEALTH_TIMEOUT) as resp:
            if resp.status == 200:
                body = json.loads(resp.read())
                models = [
                    m.get("id", "") for m in body.get("data", [])
                ]
                return True, models
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        pass
    return False, []
