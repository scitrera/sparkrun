"""Benchmarking plugin base class, spec loading, and result export."""

from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml
from scitrera_app_framework import Plugin, Variables
from vpd.next.util import read_yaml

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)

EXT_BENCHMARKING_FRAMEWORKS = "sparkrun.benchmarking"

# Keys in the benchmark: block that are NOT framework args
_KNOWN_BENCHMARK_KEYS = {"framework", "args", "metadata", "timeout"}


class BenchmarkError(Exception):
    """Raised when a benchmark spec is invalid or cannot be loaded."""


@dataclass
class BenchmarkSpec:
    """Standalone benchmark YAML definition.

    Supports two formats for the ``benchmark:`` block:

    Nested (explicit args)::

        benchmark:
          framework: llama-benchy
          args:
            pp: [2048]
            depth: [0]

    Flat (unknown keys swept into args)::

        benchmark:
          framework: llama-benchy
          pp: [2048]
          depth: [0]
    """

    source_path: str
    framework: str
    args: dict[str, Any]
    timeout: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkSpec:
        """Load and validate a benchmark YAML file.

        Supports two formats:

        1. Embedded in a recipe YAML (``benchmark:`` key wraps the block).
        2. Standalone profile file (top-level keys *are* the benchmark config).
        """
        p = Path(path)
        if not p.exists():
            raise BenchmarkError("Benchmark file not found: %s" % p)

        data = read_yaml(str(p))
        if not isinstance(data, dict):
            raise BenchmarkError("Benchmark file must contain a YAML mapping: %s" % p)

        block = data.get("benchmark")
        if not isinstance(block, dict):
            # Standalone profile file: top-level keys *are* the benchmark block
            if "framework" in data:
                block = data
            else:
                raise BenchmarkError("Benchmark file missing required 'benchmark' mapping")

        framework = block.get("framework")
        if not framework or not isinstance(framework, str):
            raise BenchmarkError("benchmark.framework is required and must be a string")

        # Explicit args take priority; unknown keys are swept in as well
        args = dict(block.get("args") or {})
        for k, v in block.items():
            if k not in _KNOWN_BENCHMARK_KEYS and k not in args:
                args[k] = v

        if not isinstance(args, dict):
            raise BenchmarkError("benchmark.args must be a mapping")

        timeout = block.get("timeout")
        if timeout is not None:
            timeout = int(timeout)

        return cls(
            source_path=str(p),
            framework=framework,
            args=args,
            timeout=timeout,
        )

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> BenchmarkSpec | None:
        """Extract benchmark config from a recipe's raw data.

        Returns None if the recipe has no ``benchmark:`` block.
        """
        block = recipe._raw.get("benchmark")
        if not isinstance(block, dict):
            return None

        framework = block.get("framework", "llama-benchy")

        # Explicit args take priority; unknown keys swept in
        args = dict(block.get("args") or {})
        for k, v in block.items():
            if k not in _KNOWN_BENCHMARK_KEYS and k not in args:
                args[k] = v

        timeout = block.get("timeout")
        if timeout is not None:
            timeout = int(timeout)

        return cls(
            source_path=recipe.source_path or "",
            framework=str(framework),
            args=args,
            timeout=timeout,
        )

    def build_command(self, extra_args: dict[str, Any] | None = None) -> list[str]:
        """Render a shell argv list for this benchmark spec.

        Command shape: ``framework --kebab-case-key VALUE ...``

        Booleans become bare flags; lists emit repeated flags.
        """
        merged_args = dict(self.args)
        if extra_args:
            merged_args.update(extra_args)

        cmd: list[str] = [self.framework]
        cmd.extend(render_args_as_flags(merged_args))

        return cmd


def render_args_as_flags(args: dict[str, Any]) -> list[str]:
    """Render a dict of args as CLI flags (--kebab-case-key value).

    Booleans become bare flags (present when True, absent when False).
    Lists emit repeated flags for each element.
    """
    parts: list[str] = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                parts.extend([flag, str(item)])
            continue
        parts.extend([flag, str(value)])
    return parts


class BenchmarkingPlugin(Plugin):
    """Abstract base for benchmarking frameworks (SAF multi-extension plugin).

    Mirrors :class:`~sparkrun.runtimes.base.RuntimePlugin` in structure.
    Subclasses register via entry points under ``sparkrun.benchmarking``.
    """

    eager = False
    framework_name: str = ""
    default_args: dict[str, Any] = {}

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.benchmarking.%s" % self.framework_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_BENCHMARKING_FRAMEWORKS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> BenchmarkingPlugin:
        return self

    # --- Framework interface ---

    @abstractmethod
    def check_prerequisites(self) -> list[str]:
        """Check and return missing prerequisites (empty list = ready).

        For example, llama-benchy checks for uvx availability.
        """
        ...

    @abstractmethod
    def build_benchmark_command(
            self,
            target_url: str,
            model: str,
            args: dict[str, Any],
            result_file: str | None = None,
    ) -> list[str]:
        """Build the benchmark command argv list.

        Args:
            target_url: The inference endpoint URL (e.g. http://host:8000/v1).
            model: Model name for the --model flag.
            args: Merged profile + CLI override args.
            result_file: Path where the framework should save results.

        Returns:
            Command argv list suitable for subprocess.
        """
        ...

    def interpret_arg(self, key: str, value: str) -> Any:
        """Interpret a CLI string arg into the correct type.

        Default: comma-separated strings become lists, else coerce_value().
        Subclasses override for framework-specific type handling.
        """
        from sparkrun.utils import coerce_value

        if "," in value:
            return [coerce_value(v.strip()) for v in value.split(",")]
        return coerce_value(value)

    @abstractmethod
    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        """Parse benchmark output into structured results dict.

        Args:
            stdout: Captured standard output.
            stderr: Captured standard error.
            result_file: Path to saved results file (if any).

        Returns:
            Structured results dict.
        """
        ...

    def get_default_args(self) -> dict[str, Any]:
        """Return default benchmark args when no profile is provided."""
        return dict(self.default_args)

    def __repr__(self) -> str:
        return "%s(framework_name=%r)" % (self.__class__.__name__, self.framework_name)


def export_results(
        *,
        recipe: Recipe,
        hosts: list[str],
        tp: int,
        cluster_id: str,
        framework_name: str,
        profile_name: str | None,
        args: dict[str, Any],
        results: dict[str, Any],
        output_path: str | Path,
) -> Path:
    """Export benchmark results to a YAML file.

    Args:
        recipe: The recipe that was benchmarked.
        hosts: Hosts used for inference.
        tp: Tensor parallelism value.
        cluster_id: Cluster ID from the inference run.
        framework_name: Name of the benchmarking framework.
        profile_name: Profile name (or None if defaults used).
        args: Benchmark args that were used.
        results: Structured results from the framework.
        output_path: Path to write the YAML file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    # noinspection PyProtectedMember
    recipe_text = yaml.safe_dump(recipe._raw, indent=2, sort_keys=True)
    recipe_hash = hashlib.sha256(recipe_text.encode("utf-8")).hexdigest()

    # Build model metadata from recipe metadata (includes auto-detected
    # values written back by Recipe.estimate_vram).
    model_meta: dict[str, Any] = {}
    if recipe.metadata.get("model_dtype"):
        model_meta["dtype"] = recipe.metadata["model_dtype"]
    if recipe.model_revision:
        model_meta["revision"] = recipe.model_revision
    if recipe.metadata.get("model_params"):
        model_meta["params"] = recipe.metadata["model_params"]
    if recipe.metadata.get("num_layers"):
        model_meta["num_layers"] = recipe.metadata["num_layers"]
    if recipe.metadata.get("num_kv_heads"):
        model_meta["num_kv_heads"] = recipe.metadata["num_kv_heads"]
    if recipe.metadata.get("head_dim"):
        model_meta["head_dim"] = recipe.metadata["head_dim"]

    data = {
        "sparkrun_benchmark": {
            "version": "1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recipe": {
                "name": recipe.name,
                "type": "sparkrun",
                "model": recipe.model,
                "container": recipe.container,
                "runtime": recipe.runtime,
                "registry": recipe.source_registry,
                "registry_git": recipe.source_registry_url or "",
                "text": recipe_text,
                "hash": recipe_hash,
            },
            "model": model_meta,
            "cluster": {
                "tp": tp,
                "cluster_id": cluster_id,
            },
            "benchmark": {
                "framework": framework_name,
                "profile": profile_name,
                "args": args,
            },
            "results": results,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Results exported to %s", output_path)
    return output_path
