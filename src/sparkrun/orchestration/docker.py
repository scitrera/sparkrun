"""Docker command string generation.

These functions are pure generators -- they produce command strings
that will be embedded into scripts and executed remotely via SSH.
They do not execute Docker commands directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)

# Standard docker run options for DGX Spark GPU workloads
_DEFAULT_DOCKER_OPTS = [
    "--privileged",
    "--gpus all",
    "--rm",
    "--ipc=host",
    "--shm-size=10.24gb",
    "--network host",
]


def docker_run_cmd(
    image: str,
    command: str = "",
    container_name: str | None = None,
    detach: bool = True,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    extra_opts: list[str] | None = None,
) -> str:
    """Generate a ``docker run`` command string.

    Args:
        image: Container image reference (e.g. ``nvcr.io/nvidia/vllm:latest``).
        command: Command to run inside the container.
        container_name: Optional ``--name`` for the container.
        detach: Run in detached mode (``-d``).
        env: Environment variables to set (``-e KEY=VALUE``).
        volumes: Volume mounts (``-v host:container``).
        extra_opts: Additional docker run options.

    Returns:
        Complete ``docker run`` command string.
    """
    parts = ["docker", "run"]

    if detach:
        parts.append("-d")

    parts.extend(_DEFAULT_DOCKER_OPTS)

    if container_name:
        parts.extend(["--name", container_name])

    if env:
        for key, value in sorted(env.items()):
            parts.extend(["-e", f"{key}={value}"])

    if volumes:
        for host_path, container_path in sorted(volumes.items()):
            parts.extend(["-v", f"{host_path}:{container_path}"])

    if extra_opts:
        parts.extend(extra_opts)

    parts.append(image)

    if command:
        parts.append(command)

    result = " ".join(parts)

    if env:
        logger.debug("docker run %s env (%d vars):", container_name or image, len(env))
        for key, value in sorted(env.items()):
            logger.debug("  %s=%s", key, value)
    logger.debug("docker run command: %s", result)

    return result


def docker_exec_cmd(
    container_name: str,
    command: str,
    detach: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Generate a ``docker exec`` command string.

    Args:
        container_name: Name of the running container.
        command: Command to execute inside the container.
        detach: Run in detached mode.
        env: Environment variables to set.

    Returns:
        Complete ``docker exec`` command string.
    """
    parts = ["docker", "exec"]
    if detach:
        parts.append("-d")
    if env:
        for key, value in sorted(env.items()):
            parts.extend(["-e", f"{key}={value}"])
    parts.extend([container_name, "bash", "-c", f"'{command}'"])
    return " ".join(parts)


def docker_stop_cmd(container_name: str, force: bool = True) -> str:
    """Generate a docker stop/rm command string.

    Args:
        container_name: Name of the container to stop.
        force: If True, use ``docker rm -f``; otherwise ``docker stop``.

    Returns:
        Command string that stops (and optionally removes) the container.
    """
    if force:
        return f"docker rm -f {container_name} 2>/dev/null || true"
    return f"docker stop {container_name} 2>/dev/null || true"


def docker_inspect_exists_cmd(image: str) -> str:
    """Generate a command to check if a docker image exists locally.

    Args:
        image: Image reference to check.

    Returns:
        Command string that exits 0 if the image exists locally.
    """
    return f"docker image inspect {image} >/dev/null 2>&1"


def docker_pull_cmd(image: str) -> str:
    """Generate a ``docker pull`` command.

    Args:
        image: Image reference to pull.

    Returns:
        Command string.
    """
    return f"docker pull {image}"


def docker_logs_cmd(
    container_name: str,
    follow: bool = False,
    tail: int | None = None,
) -> str:
    """Generate a ``docker logs`` command.

    Args:
        container_name: Name of the container.
        follow: If True, follow log output (``-f``).
        tail: Number of lines to show from the end.

    Returns:
        Command string.
    """
    parts = ["docker", "logs"]
    if follow:
        parts.append("-f")
    if tail is not None:
        parts.extend(["--tail", str(tail)])
    parts.append(container_name)
    return " ".join(parts)


def generate_cluster_id(recipe: "Recipe", hosts: list[str]) -> str:
    """Deterministic cluster identifier from recipe and host set.

    Takes the full Recipe and host list so the hash inputs can be
    expanded later (e.g. adding port, tensor_parallel) without
    changing the function signature.

    Currently hashes: runtime + model + sorted hosts.
    """
    import hashlib

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
) -> None:
    """Persist job metadata so ``cluster status`` can display recipe info.

    Writes a small YAML file to ``{cache_dir}/jobs/{hash}.yaml`` where
    *hash* is the 12-char hex portion of *cluster_id*.
    """
    from pathlib import Path
    import yaml

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
    if tp is not None:
        meta["tensor_parallel"] = int(tp)

    meta_path = jobs_dir / f"{digest}.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    logger.debug("Saved job metadata to %s", meta_path)


def load_job_metadata(cluster_id: str, cache_dir: str | None = None) -> dict | None:
    """Load job metadata for a cluster_id.  Returns ``None`` if not found."""
    from pathlib import Path
    import yaml

    if cache_dir is None:
        from sparkrun.config import DEFAULT_CACHE_DIR
        cache_dir = str(DEFAULT_CACHE_DIR)

    digest = cluster_id.removeprefix("sparkrun_")
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def generate_container_name(cluster_id: str, role: str = "head") -> str:
    """Generate a deterministic container name.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        role: Container role -- ``"head"``, ``"worker"``, or ``"solo"``.

    Returns:
        Container name in the form ``{cluster_id}_{role}``.
    """
    return f"{cluster_id}_{role}"
