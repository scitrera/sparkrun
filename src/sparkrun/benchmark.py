"""Benchmark spec loading and command rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vpd.next.util import read_yaml


class BenchmarkError(Exception):
    """Raised when a benchmark spec is invalid or cannot be loaded."""


@dataclass
class BenchmarkSpec:
    """Standalone benchmark YAML definition."""

    source_path: str
    framework: str
    args: dict[str, Any]
    recipe: str | None = None
    model: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSpec":
        """Load and validate a benchmark YAML file."""
        p = Path(path)
        if not p.exists():
            raise BenchmarkError("Benchmark file not found: %s" % p)

        data = read_yaml(str(p))
        if not isinstance(data, dict):
            raise BenchmarkError("Benchmark file must contain a YAML mapping: %s" % p)

        block = data.get("benchmark")
        if not isinstance(block, dict):
            raise BenchmarkError("Benchmark file missing required 'benchmark' mapping")

        framework = block.get("framework")
        if not framework or not isinstance(framework, str):
            raise BenchmarkError("benchmark.framework is required and must be a string")

        args = block.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise BenchmarkError("benchmark.args must be a mapping")

        recipe = data.get("recipe")
        if recipe is not None and not isinstance(recipe, str):
            raise BenchmarkError("recipe must be a string when provided")

        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise BenchmarkError("model must be a string when provided")

        return cls(
            source_path=str(p),
            framework=framework,
            args=dict(args),
            recipe=recipe,
            model=model,
        )

    def build_command(self, extra_args: dict[str, Any] | None = None) -> list[str]:
        """Render a shell argv list for this benchmark spec.

        Command shape:

        - executable: ``benchmark.framework``
        - optional metadata: ``--recipe``, ``--model``
        - args mapping rendered as ``--kebab-case-key VALUE`` pairs
          (booleans as flags; lists emit repeated flags).
        """
        merged_args = dict(self.args)
        if extra_args:
            merged_args.update(extra_args)

        cmd: list[str] = [self.framework]

        if self.recipe:
            cmd.extend(["--recipe", self.recipe])
        if self.model:
            cmd.extend(["--model", self.model])

        for key, value in merged_args.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue
            if isinstance(value, list):
                for item in value:
                    cmd.extend([flag, str(item)])
                continue
            cmd.extend([flag, str(value)])

        return cmd
