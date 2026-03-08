"""eugr builder: container image building and mod injection for eugr-vllm recipes."""

from __future__ import annotations

import logging
import subprocess
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING

from scitrera_app_framework import Variables, get_working_path

from sparkrun.builders.base import BuilderPlugin

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

EUGR_REPO_URL = "https://github.com/eugr/spark-vllm-docker.git"


class EugrBuilder(BuilderPlugin):
    """Builder for eugr-style container images with mod support.

    Handles:
    - Building container images via eugr's ``build-and-copy.sh``
    - Converting recipe ``mods`` into ``pre_exec`` entries for the
      hooks system to execute at container launch time.
    """

    builder_name = "eugr"

    _v: Variables = None
    _repo_dir: Path | None = None

    def initialize(self, v: Variables, logger_arg: Logger) -> EugrBuilder:
        """Initialize the eugr builder plugin."""
        self._v = v
        return self

    def prepare_image(
            self,
            image: str,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> str:
        """Build container image and inject mod pre_exec commands.

        If the recipe has ``build_args`` in runtime_config, builds
        the container via eugr's ``build-and-copy.sh``.  If the recipe
        has ``mods``, converts them to ``pre_exec`` entries that the
        hook system will execute at container launch time.

        Args:
            image: Target image name.
            recipe: The loaded recipe.
            hosts: Target host list (unused by builder).
            config: SparkrunConfig for cache dir resolution.
            dry_run: Show what would be done without executing.

        Returns:
            Final image name (may be unchanged).
        """
        build_args = recipe.runtime_config.get("build_args", [])
        mods = recipe.runtime_config.get("mods", [])
        has_mods = bool(mods)

        # Determine if we need to build the image.
        needs_build = bool(build_args)
        if not needs_build:
            from sparkrun.containers.registry import image_exists_locally
            if not image_exists_locally(image):
                logger.info("eugr image '%s' not found locally; will build", image)
                needs_build = True

        if not needs_build and not has_mods:
            return image  # nothing eugr-specific to prepare

        # Ensure repo is available (for build script and/or mods)
        registry_cache_root = None
        if config is not None:
            registry_cache_root = Path(config.cache_dir) / "registries"
        self._repo_dir = self.ensure_repo(registry_cache_root=registry_cache_root)

        # Build image if needed
        if needs_build:
            self._build_image(image, build_args, dry_run)

        # Convert mods to pre_exec entries
        if has_mods:
            self._inject_mod_pre_exec(recipe, mods)

        return image

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate eugr-specific recipe fields."""
        issues = []
        if not recipe.command:
            issues.append("[eugr] command template is recommended for eugr recipes")
        return issues

    # --- Mod -> pre_exec conversion ---

    def _inject_mod_pre_exec(self, recipe: Recipe, mods: list[str]) -> None:
        """Convert mod entries to pre_exec commands on the recipe.

        Each mod is a directory containing a ``run.sh`` script.  The
        conversion produces two pre_exec entries per mod:
        1. A dict with ``copy`` key — file injection via docker cp
        2. A string — execute run.sh inside the container

        Args:
            recipe: Recipe instance (pre_exec list is mutated).
            mods: List of mod directory names relative to repo root.
        """
        if not self._repo_dir:
            logger.warning("Cannot inject mods without a repo dir")
            return

        for mod_name in mods:
            mod_path = self._repo_dir / mod_name
            mod_basename = Path(mod_name).name
            dest = "/workspace/mods/%s" % mod_basename

            # Add copy entry (docker cp source into container)
            recipe.pre_exec.append({
                "copy": str(mod_path),
                "dest": dest,
            })
            # Add exec entry (run the mod script with WORKSPACE_DIR set)
            recipe.pre_exec.append(
                "export WORKSPACE_DIR=$PWD && cd %s && chmod +x run.sh && ./run.sh" % dest
            )

        logger.info("Injected %d mod(s) as pre_exec entries", len(mods))

    # --- Image building ---

    def _build_image(self, image: str, build_args: list[str], dry_run: bool = False) -> None:
        """Build container image via eugr's build-and-copy.sh.

        Args:
            image: Target image name (passed as ``-t``).
            build_args: Additional build arguments forwarded to the script.
            dry_run: Show what would be done without executing.
        """
        build_script = self._repo_dir / "build-and-copy.sh"
        if not build_script.exists():
            raise RuntimeError("build-and-copy.sh not found at %s" % build_script)

        cmd = [str(build_script), "-t", image] + build_args
        logger.info("Building eugr container: %s", " ".join(cmd))

        if dry_run:
            return

        result = subprocess.run(cmd, cwd=str(self._repo_dir))
        if result.returncode != 0:
            raise RuntimeError("eugr container build failed (exit %d)" % result.returncode)

    # --- Repo management ---

    def ensure_repo(
            self,
            cache_dir: Path | None = None,
            registry_cache_root: Path | None = None,
    ) -> Path:
        """Clone or update the eugr repo in sparkrun's cache.

        If the registry system already has a cached clone of the eugr-vllm
        repo (from recipe syncing), reuses it instead of cloning a second
        copy.  Sparse checkout is disabled on the registry clone so that
        scripts like ``build-and-copy.sh`` are available.
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
            if self._v is not None:
                cache_dir = Path(get_working_path(v=self._v)) / "cache"
            else:
                from sparkrun.core.config import resolve_cache_dir
                cache_dir = Path(resolve_cache_dir(None))
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
