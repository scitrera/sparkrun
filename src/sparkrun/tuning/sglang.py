"""SGLang fused MoE kernel tuning for DGX Spark.

Launches a container, clones SGLang benchmark scripts, and runs Triton
kernel tuning for each requested TP size.  Results are saved to the host
and auto-mounted in future ``sparkrun run`` invocations.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from sparkrun.tuning._common import (
    BaseTuner,
    DEFAULT_TP_SIZES,  # noqa: F401 — re-exported for public API
    _format_duration,  # noqa: F401 — re-exported for public API
    _get_tuning_dir,
    _get_tuning_env,
    _get_tuning_volumes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUNING_CACHE_SUBDIR = "tuning/sglang"
TUNING_CONTAINER_PATH = "/root/sglang_tuning_configs"
TUNING_CONTAINER_OUTPUT_PATH = "/tuning_output"

TUNE_CONTAINER_NAME = "sparkrun_tune"
SGLANG_CLONE_DIR = "/tmp/sglang_src"


# ---------------------------------------------------------------------------
# Host-side helpers (used by runtimes for auto-mounting)
# ---------------------------------------------------------------------------


def get_sglang_tuning_dir() -> Path:
    """Return the host-side directory for SGLang tuning configs.

    Default: ``~/.cache/sparkrun/tuning/sglang/``
    """
    return _get_tuning_dir(TUNING_CACHE_SUBDIR)


def get_sglang_tuning_volumes() -> dict[str, str] | None:
    """Return volume mapping for tuning configs if they exist.

    Returns:
        Dict mapping host dir to container dir, or ``None`` if no
        tuning configs are available.
    """
    return _get_tuning_volumes(get_sglang_tuning_dir, TUNING_CONTAINER_PATH)


def get_sglang_tuning_env() -> dict[str, str] | None:
    """Return env vars for tuning configs if they exist.

    Returns:
        Dict with ``SGLANG_MOE_CONFIG_DIR`` set, or ``None`` if no
        tuning configs are available.
    """
    return _get_tuning_env(
        get_sglang_tuning_volumes, "SGLANG_MOE_CONFIG_DIR", TUNING_CONTAINER_PATH,
    )


# ---------------------------------------------------------------------------
# SglangTuner
# ---------------------------------------------------------------------------


class SglangTuner(BaseTuner):
    """Orchestrates SGLang fused MoE kernel tuning on a single host.

    Args:
        host: Target host for tuning.
        image: Container image to use.
        model: Model name (HuggingFace repo ID).
        config: SparkrunConfig for SSH settings.
        cache_dir: HuggingFace cache directory.
        output_dir: Override for tuning output directory on host.
        skip_clone: Skip cloning SGLang repo (scripts already in image).
        dry_run: Show commands without executing.
    """

    runtime_label = "SGLang"
    container_name = TUNE_CONTAINER_NAME
    output_path = TUNING_CONTAINER_OUTPUT_PATH
    clone_script = "sglang_clone_benchmarks.sh"

    def _default_output_dir(self) -> Path:
        return get_sglang_tuning_dir()

    def _run_tune_for_tp(self, tp_size: int, triton_version: str) -> int:
        """Step 4 (per-TP): Run the tuning script for a given TP size."""
        import logging
        import time
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        logger = logging.getLogger(__name__)
        t0 = time.monotonic()

        tune_cmd = build_tuning_command(self.model, tp_size, triton_version=triton_version)

        exec_cmd = docker_exec_cmd(self.container_name, tune_cmd)

        # Tuning can take many hours (e.g. 4+ hours for TP=4 on large
        # models).  Use an 8-hour timeout so remote SSH sessions aren't
        # killed prematurely.
        result = run_command_on_host(
            self.host, exec_cmd,
            ssh_kwargs=self.ssh_kwargs, timeout=28800, dry_run=self.dry_run,
        )

        if self.dry_run:
            logger.info("  [dry-run] Would run tuning for TP=%d", tp_size)
            return 0

        elapsed = time.monotonic() - t0
        if not result.success:
            logger.error(
                "  Tuning for TP=%d failed (exit %d, %.1fs): %s",
                tp_size, result.returncode, elapsed, result.stderr[:300],
            )
            return result.returncode

        logger.info("  TP=%d tuning complete (%.1fs)", tp_size, elapsed)
        return 0


def build_tuning_command(model: str, tp_size: int, triton_version: str | None = None) -> str:
    """Build the tuning command string for display/testing.

    Args:
        model: Model name.
        tp_size: Tensor parallel size.
        triton_version: Triton version string (e.g. ``"3.6.0"``).  When
            provided, the versioned config subdirectory is pre-created so
            SGLang's ``save_configs()`` can write into it (it doesn't create
            directories itself).

    Returns:
        The tuning command string.

    Note:
        SGLang's ``get_config_file_name()`` returns only the basename
        (e.g. ``E=64,N=2560,...json``).  The benchmark ``save_configs()``
        writes to that basename via ``open(filename, "w")``, which
        resolves relative to the working directory.  We therefore ``cd``
        into the versioned output subdirectory so the JSON files land in
        the mounted volume rather than being lost inside the container.

        ``SGLANG_MOE_CONFIG_DIR`` is still set for any runtime code that
        reads configs via the standard loader.
    """
    config_dir = TUNING_CONTAINER_OUTPUT_PATH
    # Build the versioned output subdirectory path.  save_configs() writes
    # to CWD, so we cd into this directory before running the script.
    if triton_version and triton_version != "unknown":
        versioned = "triton_%s" % triton_version.replace(".", "_")
        output_subdir = "%s/configs/%s" % (config_dir, versioned)
    else:
        output_subdir = "%s/configs" % config_dir
    return (
        "mkdir -p %s && "
        "cd %s && "
        "SGLANG_MOE_CONFIG_DIR=%s "
        "python3 %s/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py "
        "--model %s --tp-size %d --tune"
    ) % (output_subdir, output_subdir, config_dir, SGLANG_CLONE_DIR, shlex.quote(model), tp_size)
