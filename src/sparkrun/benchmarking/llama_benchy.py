"""llama-benchy benchmarking framework plugin for sparkrun."""

from __future__ import annotations

import csv
import io
import logging
import shutil
from logging import Logger
from typing import Any

from scitrera_app_framework import Variables

from sparkrun.benchmarking.base import BenchmarkingPlugin

logger = logging.getLogger(__name__)

# llama-benchy args that accept multiple space-separated values on CLI,
# stored as comma-separated strings in profiles.
_LIST_ARGS = {"pp", "tg", "depth", "concurrency"}

# llama-benchy boolean flags (present = enabled, absent = disabled).
_BOOL_ARGS = {
    "no_cache", "enable_prefix_caching", "no_warmup", "skip_coherence",
    "adapt_prompt", "no_adapt_prompt",
    "save_total_throughput_timeseries", "save_all_throughput_timeseries",
}

# Common shorthand aliases → canonical llama-benchy arg names.
# Profiles may use shorter key names for convenience.
_ARG_ALIASES: dict[str, str] = {
    "prefix_caching": "enable_prefix_caching",
}


class LlamaBenchyFramework(BenchmarkingPlugin):
    """llama-benchy benchmarking framework.

    Uses ``uvx llama-benchy`` to run OpenAI-compatible inference benchmarks.
    Produces CSV output for machine-parseable results.
    """

    framework_name = "llama-benchy"
    default_args: dict[str, Any] = {
        "pp": [2048],
        "depth": [0],
        "enable_prefix_caching": True,
    }

    def initialize(self, v: Variables, logger_arg: Logger) -> LlamaBenchyFramework:
        return self

    def check_prerequisites(self) -> list[str]:
        """Check that uvx is available on PATH."""
        missing = []
        if shutil.which("uvx") is None:
            missing.append(
                "uvx not found on PATH. Install uv: "
                "https://docs.astral.sh/uv/getting-started/installation/"
            )
        return missing

    def build_benchmark_command(
            self,
            target_url: str,
            model: str,
            args: dict[str, Any],
            result_file: str | None = None,
    ) -> list[str]:
        """Build the uvx llama-benchy command.

        Always uses ``--format csv`` for machine-parseable output and
        ``--save-result`` to capture results to a file.
        """
        cmd = [
            "uvx", "llama-benchy",
            "--base-url", target_url,
            "--model", model,
            "--format", "csv",
        ]

        if result_file:
            cmd.extend(["--save-result", result_file])

        # Render args as CLI flags
        for key, value in args.items():
            # Skip args we handle explicitly above
            if key in ("base_url", "model", "format", "save_result"):
                continue

            # Resolve shorthand aliases to canonical names
            key = _ARG_ALIASES.get(key, key)

            flag = "--" + key.replace("_", "-")

            if key in _BOOL_ARGS or isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue

            if isinstance(value, list):
                # llama-benchy takes space-separated values after the flag
                cmd.append(flag)
                for item in value:
                    cmd.append(str(item))
                continue

            cmd.extend([flag, str(value)])

        return cmd

    def interpret_arg(self, key: str, value: str) -> Any:
        """Interpret a CLI string arg into the correct type.

        Known list args (pp, tg, depth, concurrency) become lists when
        comma-separated. Known booleans become bool. Others use coerce_value().
        """
        from sparkrun.utils import coerce_value

        # Resolve shorthand aliases
        key = _ARG_ALIASES.get(key, key)

        if key in _BOOL_ARGS:
            return coerce_value(value)

        if key in _LIST_ARGS or "," in value:
            return [coerce_value(v.strip()) for v in value.split(",")]

        return coerce_value(value)

    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        """Parse llama-benchy CSV output into structured results.

        Reads from the ``--save-result`` file if available, otherwise
        falls back to parsing stdout.
        """
        rows: list[dict[str, str]] = []

        # Try reading from saved result file first
        csv_text = None
        if result_file:
            try:
                from pathlib import Path
                csv_text = Path(result_file).read_text()
            except (OSError, FileNotFoundError):
                logger.debug("Could not read result file %s, falling back to stdout", result_file)

        if csv_text is None:
            csv_text = stdout

        # Parse CSV content
        if csv_text.strip():
            try:
                reader = csv.DictReader(io.StringIO(csv_text))
                rows = list(reader)
            except csv.Error:
                logger.warning("Failed to parse CSV output from llama-benchy")

        return {
            "rows": rows,
            'csv': csv_text,
            "stdout": stdout,
        }
