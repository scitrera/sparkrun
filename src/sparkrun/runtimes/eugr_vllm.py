"""eugr-vllm runtime: full delegation to eugr/spark-vllm-docker scripts."""

from __future__ import annotations

import logging
import subprocess
from logging import Logger
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml
from scitrera_app_framework import Variables, get_working_path

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)

EUGR_REPO_URL = "https://github.com/eugr/spark-vllm-docker.git"


class EugrVllmRuntime(RuntimePlugin):
    """Full delegation to eugr/spark-vllm-docker.

    This runtime clones the eugr repo and calls their scripts directly.
    Mods, local builds, and all eugr-specific features work natively
    because we're running their code, not reimplementing it.
    """

    _v: Variables = None
    runtime_name = "eugr-vllm"
    default_image_prefix = ""  # eugr uses local builds

    def initialize(self, v: Variables, logger_arg: Logger) -> EugrVllmRuntime:
        """Initialize the eugr-vllm runtime plugin."""
        self._v = v
        return self

    def is_delegating_runtime(self) -> bool:
        """eugr-vllm delegates entirely to external scripts."""
        return True

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve container -- eugr manages its own container builds."""
        return recipe.container or "vllm-node-tf5"

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None) -> str:
        """Generate command -- eugr scripts handle command generation internally."""
        return recipe.command or ""

    def ensure_repo(
        self,
        cache_dir: Path | None = None,
        registry_cache_root: Path | None = None,
    ) -> Path:
        """Clone or update the eugr repo in sparkrun's cache.

        If the registry system already has a cached clone of the eugr-vllm
        repo (from recipe syncing), reuses it instead of cloning a second
        copy.  Sparse checkout is disabled on the registry clone so that
        scripts like ``run-recipe.sh`` are available.
        """
        # Check if registry already has this repo cloned
        if registry_cache_root is not None:
            registry_repo = registry_cache_root / "eugr-vllm"
            if (registry_repo / ".git").exists():
                logger.info("Reusing eugr repo from registry cache: %s", registry_repo)
                self._ensure_full_checkout(registry_repo)
                self._update_repo(registry_repo)
                return registry_repo

        if cache_dir is None:
            cache_dir = Path(get_working_path(v=self._v)) / "cache"
        repo_dir = cache_dir / "eugr-spark-vllm-docker"

        if repo_dir.exists() and (repo_dir / ".git").exists():
            self._update_repo(repo_dir)
        else:
            logger.info("Cloning eugr/spark-vllm-docker...")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", EUGR_REPO_URL, str(repo_dir)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to clone eugr repo: %s" % result.stderr.strip()
                )

        return repo_dir

    @staticmethod
    def _ensure_full_checkout(repo_dir: Path) -> None:
        """Disable sparse checkout so all repo files are available."""
        sparse_file = repo_dir / ".git" / "info" / "sparse-checkout"
        if not sparse_file.exists():
            return  # not sparse, nothing to do
        logger.debug("Disabling sparse checkout on %s", repo_dir)
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "disable"],
            capture_output=True, text=True,
        )

    @staticmethod
    def _update_repo(repo_dir: Path) -> None:
        """Pull latest changes for an existing repo clone."""
        logger.info("Updating eugr/spark-vllm-docker repo...")
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to update eugr repo (continuing with existing): %s",
                result.stderr.strip(),
            )

    def write_eugr_recipe(self, recipe: Recipe, repo_dir: Path) -> Path:
        """Convert sparkrun v2 recipe back to eugr v1 format."""
        eugr_data = {
            "recipe_version": "1",
            "name": recipe.name,
            "description": recipe.description,
            "model": recipe.model,
            "container": recipe.container or "vllm-node-tf5",
            "cluster_only": recipe.min_nodes > 1,
            "solo_only": recipe.max_nodes == 1 if recipe.max_nodes else False,
            "build_args": recipe.runtime_config.get("build_args", []),
            "mods": recipe.runtime_config.get("mods", []),
            "defaults": recipe.defaults,
            "env": recipe.env,
        }
        if recipe.command:
            eugr_data["command"] = recipe.command

        recipes_dir = repo_dir / "recipes"
        recipes_dir.mkdir(exist_ok=True)
        out_path = recipes_dir / ("_sparkrun_%s.yaml" % recipe.slug)
        with open(out_path, "w") as f:
            yaml.dump(eugr_data, f, default_flow_style=False, sort_keys=False)

        return out_path

    def run_delegated(self, recipe: Recipe, overrides: dict[str, Any],
                      hosts: list[str] | None = None, solo: bool = False,
                      setup: bool = False, dry_run: bool = False,
                      cache_dir: Path | None = None,
                      registry_cache_root: Path | None = None) -> int:
        """Delegate entirely to eugr's run-recipe.sh.

        Returns the process exit code.
        """
        repo_dir = self.ensure_repo(cache_dir, registry_cache_root=registry_cache_root)
        recipe_path = self.write_eugr_recipe(recipe, repo_dir)

        run_script = repo_dir / "run-recipe.sh"
        if not run_script.exists():
            raise RuntimeError("eugr run-recipe.sh not found at %s" % run_script)

        cmd = [str(run_script), str(recipe_path)]

        if solo:
            cmd.append("--solo")
        if setup:
            cmd.append("--setup")
        if dry_run:
            cmd.append("--dry-run")
        if hosts:
            cmd.extend(["-n", ",".join(hosts)])

        # Pass overrides as CLI flags
        for key, value in overrides.items():
            flag = key.replace("_", "-")
            cmd.extend(["--%s" % flag, str(value)])

        logger.info("Delegating to eugr: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(repo_dir))
        return result.returncode

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate eugr-vllm-specific recipe fields."""
        issues = super().validate_recipe(recipe)
        if not recipe.command:
            issues.append("[eugr-vllm] command template is recommended for eugr recipes")
        return issues

    # --- Log following ---

    def follow_logs(
            self,
            hosts: list[str],
            cluster_id: str = "sparkrun0",
            config=None,
            dry_run: bool = False,
            tail: int = 100,
    ) -> None:
        """No-op — eugr-vllm delegates to external scripts."""
        pass

    # --- Launch / Stop ---

    def run(
            self,
            hosts: list[str],
            image: str,
            serve_command: str,
            recipe: Recipe,
            overrides: dict[str, Any],
            *,
            cluster_id: str = "sparkrun0",
            env: dict[str, str] | None = None,
            cache_dir: str | None = None,
            config=None,
            dry_run: bool = False,
            detached: bool = True,
            skip_ib_detect: bool = False,
            setup: bool = False,
            **kwargs,
    ) -> int:
        """Launch via eugr delegation.

        Wraps :meth:`run_delegated` with the standard runtime interface.
        """
        # Derive registry cache root from config so ensure_repo() can
        # reuse an existing registry clone instead of double-cloning.
        registry_cache_root = None
        if config is not None:
            registry_cache_root = Path(config.cache_dir) / "registries"

        is_solo = len(hosts) <= 1
        return self.run_delegated(
            recipe=recipe,
            overrides=overrides,
            hosts=hosts if not is_solo else None,
            solo=is_solo,
            setup=setup,
            dry_run=dry_run,
            cache_dir=Path(cache_dir) if cache_dir else None,
            registry_cache_root=registry_cache_root,
        )
