"""Git-based recipe registry system for sparkrun.

This module provides a registry manager that tracks and syncs recipe collections
from remote git repositories using sparse checkouts for efficiency.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from vpd.next.util import read_yaml

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised for registry-specific errors."""

    pass


@dataclass
class RegistryEntry:
    """Represents a recipe registry source.

    Attributes:
        name: Unique identifier for the registry
        url: Git repository URL
        subpath: Path within the repository containing recipes
        description: Human-readable description
        enabled: Whether this registry is active
    """

    name: str
    url: str
    subpath: str
    description: str = ""
    enabled: bool = True


DEFAULT_REGISTRIES = [
    RegistryEntry(
        name="sparkrun-official",
        url="https://github.com/scitrera/oss-spark-run",
        subpath="recipes",
        description="Official sparkrun recipes",
        enabled=True,
    ),
    RegistryEntry(
        name="eugr-vllm",
        url="https://github.com/eugr/spark-vllm-docker",
        subpath="recipes",
        description="EUGR vLLM recipes for DGX Spark",
        enabled=True,
    ),
]


class RegistryManager:
    """Manages recipe registries with git-based syncing.

    The manager tracks registry configurations, handles shallow git clones
    with sparse checkouts, and provides recipe discovery across all registries.
    """

    def __init__(self, config_root: Path, cache_root: Path | None = None) -> None:
        """Initialize the registry manager.

        Args:
            config_root: Directory containing registries.yaml
            cache_root: Optional cache directory, defaults to ~/.cache/sparkrun/registries
        """
        self.config_root = Path(config_root)
        self.cache_root = (
            Path(cache_root) if cache_root else Path.home() / ".cache/sparkrun/registries"
        )
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @property
    def _registries_path(self) -> Path:
        """Path to the registries configuration file."""
        return self.config_root / "registries.yaml"

    def _cache_dir(self, name: str) -> Path:
        """Get the cache directory for a specific registry.

        Args:
            name: Registry name

        Returns:
            Path to the cache directory
        """
        return self.cache_root / name

    def _recipe_dir(self, entry: RegistryEntry) -> Path | None:
        """Get the recipe directory within a cached registry.

        Args:
            entry: Registry entry

        Returns:
            Path to the recipe directory, or None if not cached
        """
        cache_dir = self._cache_dir(entry.name)
        recipe_path = cache_dir / entry.subpath
        return recipe_path if recipe_path.exists() else None

    def _load_registries(self) -> list[RegistryEntry]:
        """Load registries from YAML configuration.

        Returns:
            List of registry entries, or DEFAULT_REGISTRIES if config doesn't exist
        """
        if not self._registries_path.exists():
            logger.debug("No registries.yaml found, using defaults")
            return list(DEFAULT_REGISTRIES)

        try:
            data = read_yaml(self._registries_path)
            registries = data.get("registries", [])
            return [
                RegistryEntry(
                    name=r["name"],
                    url=r["url"],
                    subpath=r["subpath"],
                    description=r.get("description", ""),
                    enabled=r.get("enabled", True),
                )
                for r in registries
            ]
        except Exception as e:
            logger.warning("Failed to load registries.yaml: %s", e)
            return list(DEFAULT_REGISTRIES)

    def _save_registries(self, entries: list[RegistryEntry]) -> None:
        """Save registries to YAML configuration.

        Args:
            entries: List of registry entries to save
        """
        data = {
            "registries": [
                {
                    "name": e.name,
                    "url": e.url,
                    "subpath": e.subpath,
                    "description": e.description,
                    "enabled": e.enabled,
                }
                for e in entries
            ]
        }
        with open(self._registries_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.debug("Saved registries to %s", self._registries_path)

    @staticmethod
    def _git_env() -> dict[str, str]:
        """Return environment variables for non-interactive git operations."""
        import os
        env = os.environ.copy()
        # Prevent git from prompting for credentials — fail immediately instead
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def _clone_or_pull(self, entry: RegistryEntry) -> bool:
        """Clone or update a registry repository.

        Uses shallow clone with sparse checkout for efficiency. Git command
        failures are logged but not raised (best-effort sync).

        Args:
            entry: Registry entry to sync

        Returns:
            True if the operation succeeded, False otherwise.
        """
        cache_dir = self._cache_dir(entry.name)
        git_env = self._git_env()

        try:
            if (cache_dir / ".git").exists():
                # Update existing repository
                logger.debug("Updating registry %s", entry.name)
                result = subprocess.run(
                    ["git", "-C", str(cache_dir), "pull", "--ff-only"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug(
                        "Git pull failed for %s: %s", entry.name, result.stderr
                    )
                    return False
            else:
                # Fresh clone with sparse checkout
                logger.debug("Cloning registry %s", entry.name)
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Shallow clone with blob filtering
                result = subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--filter=blob:none",
                        "--sparse",
                        entry.url,
                        str(cache_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug(
                        "Git clone failed for %s: %s", entry.name, result.stderr
                    )
                    return False

                # Configure sparse checkout for subpath only
                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(cache_dir),
                        "sparse-checkout",
                        "set",
                        entry.subpath,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug(
                        "Sparse checkout setup failed for %s: %s",
                        entry.name,
                        result.stderr,
                    )
                    return False
        except subprocess.TimeoutExpired:
            logger.debug("Git operation timed out for %s", entry.name)
            return False
        except Exception as e:
            logger.debug("Failed to sync registry %s: %s", entry.name, e)
            return False

        return True

    def add_registry(self, entry: RegistryEntry) -> None:
        """Add a new registry.

        Args:
            entry: Registry entry to add

        Raises:
            RegistryError: If a registry with the same name already exists
        """
        registries = self._load_registries()
        if any(r.name == entry.name for r in registries):
            raise RegistryError(f"Registry {entry.name!r} already exists")
        registries.append(entry)
        self._save_registries(registries)
        logger.info("Added registry %s", entry.name)

    def remove_registry(self, name: str) -> None:
        """Remove a registry by name.

        Args:
            name: Registry name to remove

        Raises:
            RegistryError: If the registry is not found
        """
        registries = self._load_registries()
        filtered = [r for r in registries if r.name != name]
        if len(filtered) == len(registries):
            raise RegistryError(f"Registry {name!r} not found")
        self._save_registries(filtered)
        logger.info("Removed registry %s", name)

    def list_registries(self) -> list[RegistryEntry]:
        """List all configured registries.

        Returns:
            List of all registry entries
        """
        return self._load_registries()

    def get_registry(self, name: str) -> RegistryEntry:
        """Get a single registry by name.

        Args:
            name: Registry name

        Returns:
            Registry entry

        Raises:
            RegistryError: If the registry is not found
        """
        registries = self._load_registries()
        for entry in registries:
            if entry.name == name:
                return entry
        raise RegistryError(f"Registry {name!r} not found")

    def update(
            self,
            name: str | None = None,
            progress: Callable[[str, bool], None] | None = None,
    ) -> dict[str, bool]:
        """Update one or all registries.

        Performs shallow clone or pull for specified registry or all enabled
        registries if name is None.

        Args:
            name: Optional registry name to update, or None for all.
            progress: Optional callback invoked after each registry with
                ``(registry_name, success)``.

        Returns:
            Mapping of registry name to success status for each registry
            that was attempted.
        """
        registries = self._load_registries()
        results: dict[str, bool] = {}

        if name is not None:
            # Update single registry
            entry = self.get_registry(name)
            if entry.enabled:
                ok = self._clone_or_pull(entry)
                results[entry.name] = ok
                if progress:
                    progress(entry.name, ok)
            else:
                logger.warning("Registry %s is disabled, skipping update", name)
                results[entry.name] = False
                if progress:
                    progress(entry.name, False)
        else:
            # Update all enabled registries
            for entry in registries:
                if entry.enabled:
                    ok = self._clone_or_pull(entry)
                    results[entry.name] = ok
                    if progress:
                        progress(entry.name, ok)

        return results

    def ensure_initialized(self) -> None:
        """Ensure registries are initialized.

        If no cache exists, runs update() to perform initial sync.
        """
        registries = self._load_registries()
        needs_init = False

        for entry in registries:
            if entry.enabled:
                cache_dir = self._cache_dir(entry.name)
                if not (cache_dir / ".git").exists():
                    needs_init = True
                    break

        if needs_init:
            logger.debug("Initializing registries")
            self.update()

    def get_recipe_paths(self) -> list[Path]:
        """Get all recipe directories from cached registries.

        Returns:
            List of paths to recipe directories (only from enabled registries)
        """
        paths = []
        registries = self._load_registries()

        for entry in registries:
            if not entry.enabled:
                continue

            recipe_dir = self._recipe_dir(entry)
            if recipe_dir:
                paths.append(recipe_dir)
            else:
                logger.debug(
                    "Registry %s not cached or recipe path not found", entry.name
                )

        return paths

    def _list_dir_recipes(self, recipe_dir: Path, registry_name: str) -> list[dict[str, Any]]:
        """List all recipes in a directory with metadata.

        Args:
            recipe_dir: Directory to scan for .yaml recipe files.
            registry_name: Name of the registry this directory belongs to.

        Returns:
            List of recipe metadata dicts.
        """
        recipes = []
        if not recipe_dir.is_dir():
            return recipes

        from sparkrun.recipe import resolve_runtime

        for f in sorted(recipe_dir.rglob("*.yaml")):
            stem = f.stem
            try:
                data = read_yaml(str(f))
                if not isinstance(data, dict):
                    continue
                defaults = data.get("defaults", {})
                recipes.append({
                    "name": data.get("name", stem),
                    "file": stem,
                    "path": str(f),
                    "model": data.get("model", ""),
                    "description": data.get("description", ""),
                    "runtime": resolve_runtime(data),
                    "registry": registry_name,
                    "min_nodes": data.get("min_nodes", 1),
                    "tp": defaults.get("tensor_parallel", "") if isinstance(defaults, dict) else "",
                    "gpu_mem": defaults.get("gpu_memory_utilization", "") if isinstance(defaults, dict) else "",
                })
            except Exception:
                logger.debug("Skipping invalid recipe file: %s", f)
        return recipes

    def search_recipes(self, query: str) -> list[dict[str, Any]]:
        """Search for recipes across all registries.

        Performs case-insensitive substring matching on recipe name, file stem,
        model, and description fields.

        Args:
            query: Search query string

        Returns:
            List of recipe metadata dicts with 'registry' field added
        """
        results = []
        query_lower = query.lower()
        registries = self._load_registries()

        for entry in registries:
            if not entry.enabled:
                continue
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir is None:
                continue
            for recipe in self._list_dir_recipes(recipe_dir, entry.name):
                searchable = [
                    recipe.get("name", "").lower(),
                    recipe.get("file", "").lower(),
                    recipe.get("model", "").lower(),
                    recipe.get("description", "").lower(),
                ]
                if any(query_lower in s for s in searchable):
                    results.append(recipe)

        return results

    def registry_for_path(self, path: Path) -> str | None:
        """Return the registry name that owns the given path, or None."""
        registries = self._load_registries()
        for entry in registries:
            if not entry.enabled:
                continue
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir and path.is_relative_to(recipe_dir):
                return entry.name
        return None

    def find_recipe_in_registries(self, name: str) -> list[tuple[str, Path]]:
        """Find a recipe by file stem across all registries.

        Searches for recipes whose file stem matches the given name.

        Args:
            name: Recipe file stem to find (e.g. 'glm-4.7-flash-awq')

        Returns:
            List of (registry_name, recipe_path) tuples for disambiguation
        """
        matches = []
        registries = self._load_registries()

        for entry in registries:
            if not entry.enabled:
                continue
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir is None:
                continue
            # Flat lookup first (existing behavior)
            for ext in (".yaml", ".yml"):
                candidate = recipe_dir / (name + ext)
                if candidate.exists():
                    matches.append((entry.name, candidate))

        # If flat lookup found nothing, search subdirectories by stem
        if not matches:
            for entry in registries:
                if not entry.enabled:
                    continue
                recipe_dir = self._recipe_dir(entry)
                if recipe_dir is None:
                    continue
                for ext in (".yaml", ".yml"):
                    for candidate in sorted(recipe_dir.rglob(f"{name}{ext}")):
                        matches.append((entry.name, candidate))

        return matches
