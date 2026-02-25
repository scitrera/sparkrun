"""SGLang fused MoE kernel tuning for DGX Spark.

Launches a container, clones SGLang benchmark scripts, and runs Triton
kernel tuning for each requested TP size.  Results are saved to the host
and auto-mounted in future ``sparkrun run`` invocations.
"""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

from sparkrun.tuning import (
    get_sglang_tuning_dir,
    TUNING_CONTAINER_OUTPUT_PATH,
)

if TYPE_CHECKING:
    from sparkrun.config import SparkrunConfig

logger = logging.getLogger(__name__)

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Returns ``"Xs"`` for durations under 60s, ``"Xm Ys"`` for durations
    under an hour, and ``"Xh Ym Zs"`` for longer durations.
    """
    s = int(seconds)
    if s < 60:
        return "%.1fs" % seconds
    m, s = divmod(s, 60)
    if m < 60:
        return "%dm %ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %dm %ds" % (h, m, s)


TUNE_CONTAINER_NAME = "sparkrun_tune"
SGLANG_CLONE_DIR = "/tmp/sglang_src"
DEFAULT_TP_SIZES = (1, 2, 4, 8)


class SglangTuner:
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

    def __init__(
            self,
            host: str,
            image: str,
            model: str,
            config: SparkrunConfig | None = None,
            cache_dir: str | None = None,
            output_dir: str | None = None,
            skip_clone: bool = False,
            dry_run: bool = False,
    ):
        self.host = host
        self.image = image
        self.model = model
        self.config = config
        self.cache_dir = cache_dir
        self.output_dir = output_dir or str(get_sglang_tuning_dir())
        self.skip_clone = skip_clone
        self.dry_run = dry_run

        from sparkrun.orchestration.primitives import build_ssh_kwargs
        self.ssh_kwargs = build_ssh_kwargs(config)

    def run_tuning(
            self,
            tp_sizes: tuple[int, ...] = DEFAULT_TP_SIZES,
            parallel: int = 1,
    ) -> int:
        """Run the full tuning flow.

        Args:
            tp_sizes: Tensor parallel sizes to tune for.
            parallel: Max concurrent tuning jobs (1 = sequential).

        Returns:
            Exit code (0 = success).
        """
        import time

        logger.info("=" * 60)
        logger.info("sparkrun SGLang Kernel Tuner")
        logger.info("=" * 60)
        logger.info("Host:       %s", self.host)
        logger.info("Image:      %s", self.image)
        logger.info("Model:      %s", self.model)
        logger.info("TP sizes:   %s", ", ".join(str(t) for t in tp_sizes))
        logger.info("Parallel:   %d", parallel)
        logger.info("Output:     %s", self.output_dir)
        logger.info("Mode:       %s", "DRY-RUN" if self.dry_run else "LIVE")
        logger.info("=" * 60)

        t_total = time.monotonic()
        tp_timings: list[tuple[int, float]] = []  # (tp_size, seconds)

        try:
            # Step 1: Launch container
            rc = self._launch_container()
            if rc != 0:
                return rc

            # Step 2: Clone benchmark scripts
            if not self.skip_clone:
                rc = self._clone_benchmarks()
                if rc != 0:
                    return rc
            else:
                logger.info("Step 2/5: Skipping clone (--skip-clone)")

            # Step 3: Detect Triton version
            triton_version = self._detect_triton_version()

            # Step 4: Run tuning for each TP size
            if parallel > 1 and len(tp_sizes) > 1:
                rc = self._run_tuning_parallel(
                    tp_sizes, triton_version, parallel, tp_timings,
                )
            else:
                rc = self._run_tuning_sequential(
                    tp_sizes, triton_version, tp_timings,
                )

            if rc != 0:
                return rc

            logger.info("Step 5/5: Tuning complete!")
            total_elapsed = time.monotonic() - t_total
            self._print_timing_summary(tp_timings, total_elapsed)
            return 0

        finally:
            self._cleanup_container()

    def _run_tuning_sequential(
            self,
            tp_sizes: tuple[int, ...],
            triton_version: str,
            tp_timings: list[tuple[int, float]],
    ) -> int:
        """Run tuning for each TP size sequentially."""
        import time
        for i, tp in enumerate(tp_sizes):
            logger.info(
                "Step 4/5: Tuning TP=%d (%d/%d)...",
                tp, i + 1, len(tp_sizes),
            )
            t_tp = time.monotonic()
            rc = self._run_tune_for_tp(tp, triton_version)
            tp_timings.append((tp, time.monotonic() - t_tp))
            if rc != 0:
                logger.error("Tuning failed for TP=%d (exit %d)", tp, rc)
                return rc
        return 0

    def _run_tuning_parallel(
            self,
            tp_sizes: tuple[int, ...],
            triton_version: str,
            max_workers: int,
            tp_timings: list[tuple[int, float]],
    ) -> int:
        """Run tuning for TP sizes in parallel batches."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        effective_workers = min(max_workers, len(tp_sizes))
        logger.info(
            "Step 4/5: Tuning %d TP sizes with %d parallel workers...",
            len(tp_sizes), effective_workers,
        )

        failed: list[tuple[int, int]] = []  # (tp_size, exit_code)

        def _tune_one(tp: int) -> tuple[int, int, float]:
            t0 = time.monotonic()
            rc = self._run_tune_for_tp(tp, triton_version)
            return tp, rc, time.monotonic() - t0

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_tune_one, tp): tp
                for tp in tp_sizes
            }
            for future in as_completed(futures):
                tp, rc, elapsed = future.result()
                tp_timings.append((tp, elapsed))
                if rc != 0:
                    logger.error("Tuning failed for TP=%d (exit %d)", tp, rc)
                    failed.append((tp, rc))
                else:
                    logger.info("  TP=%d done (%s)", tp, _format_duration(elapsed))

        # Sort timings by TP size for consistent display
        tp_timings.sort(key=lambda x: x[0])

        if failed:
            logger.error(
                "Tuning failed for TP size(s): %s",
                ", ".join(str(tp) for tp, _ in failed),
            )
            return failed[0][1]
        return 0

    def _launch_container(self) -> int:
        """Step 1: Launch a tuning container with sleep infinity."""
        import time
        from sparkrun.orchestration.primitives import build_volumes, run_script_on_host
        from sparkrun.orchestration.scripts import generate_container_launch_script

        t0 = time.monotonic()
        logger.info("Step 1/5: Launching tuning container on %s...", self.host)

        volumes = build_volumes(self.cache_dir)
        # Mount tuning output directory
        volumes[self.output_dir] = TUNING_CONTAINER_OUTPUT_PATH

        launch_script = generate_container_launch_script(
            image=self.image,
            container_name=TUNE_CONTAINER_NAME,
            command="sleep infinity",
            volumes=volumes,
        )

        result = run_script_on_host(
            self.host, launch_script,
            ssh_kwargs=self.ssh_kwargs, timeout=120, dry_run=self.dry_run,
        )

        if not result.success and not self.dry_run:
            logger.error("Failed to launch tuning container: %s", result.stderr)
            return 1

        logger.info("Step 1/5: Container launched (%.1fs)", time.monotonic() - t0)
        return 0

    def _clone_benchmarks(self) -> int:
        """Step 2: Clone SGLang benchmark scripts inside the container."""
        import time
        from sparkrun.orchestration.primitives import run_script_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd
        from sparkrun.scripts import read_script

        t0 = time.monotonic()
        logger.info("Step 2/5: Cloning SGLang benchmark scripts...")

        clone_script = read_script("sglang_clone_benchmarks.sh")
        exec_cmd = docker_exec_cmd(TUNE_CONTAINER_NAME, clone_script)

        # Wrap in a bash script for run_script_on_host
        script = "#!/bin/bash\nset -uo pipefail\n%s\n" % exec_cmd

        result = run_script_on_host(
            self.host, script,
            ssh_kwargs=self.ssh_kwargs, timeout=120, dry_run=self.dry_run,
        )

        if not result.success and not self.dry_run:
            logger.error("Failed to clone benchmark scripts: %s", result.stderr)
            return 1

        logger.info("Step 2/5: Clone done (%.1fs)", time.monotonic() - t0)
        return 0

    def _detect_triton_version(self) -> str:
        """Step 3: Detect Triton version inside the container."""
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        logger.info("Step 3/5: Detecting Triton version...")

        detect_cmd = docker_exec_cmd(
            TUNE_CONTAINER_NAME,
            "python3 -c \"import triton; print(triton.__version__)\"",
        )

        result = run_command_on_host(
            self.host, detect_cmd,
            ssh_kwargs=self.ssh_kwargs, timeout=30, dry_run=self.dry_run,
        )

        if self.dry_run:
            logger.info("Step 3/5: [dry-run] Would detect Triton version")
            return "unknown"

        version = "unknown"
        if result.success and result.stdout.strip():
            version = result.stdout.strip().splitlines()[-1].strip()
            logger.info("Step 3/5: Triton version: %s", version)
        else:
            logger.warning(
                "Step 3/5: Could not detect Triton version, using 'unknown': %s",
                result.stderr[:200] if result.stderr else "(no output)",
            )

        return version

    def _run_tune_for_tp(self, tp_size: int, triton_version: str) -> int:
        """Step 4 (per-TP): Run the tuning script for a given TP size."""
        import time
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        t0 = time.monotonic()

        tune_cmd = build_tuning_command(self.model, tp_size, triton_version=triton_version)

        exec_cmd = docker_exec_cmd(TUNE_CONTAINER_NAME, tune_cmd)

        # Use a long timeout — tuning can take many minutes
        result = run_command_on_host(
            self.host, exec_cmd,
            ssh_kwargs=self.ssh_kwargs, timeout=3600, dry_run=self.dry_run,
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

    def _print_timing_summary(
            self,
            tp_timings: list[tuple[int, float]],
            total_elapsed: float,
    ) -> None:
        """Print a timing summary table after tuning completes."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Tuning Summary")
        logger.info("=" * 60)
        logger.info("  %-8s  %s", "TP Size", "Duration")
        logger.info("  %-8s  %s", "-------", "--------")
        for tp, elapsed in tp_timings:
            logger.info("  %-8d  %s", tp, _format_duration(elapsed))
        logger.info("  %-8s  %s", "-------", "--------")
        logger.info("  %-8s  %s", "Total", _format_duration(total_elapsed))
        logger.info("")
        logger.info("Tuning configs saved to: %s", self.output_dir)
        logger.info(
            "These will be auto-mounted in future 'sparkrun run' "
            "invocations for SGLang recipes."
        )
        logger.info("=" * 60)

    def _cleanup_container(self) -> None:
        """Step 5: Remove the tuning container."""
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_stop_cmd

        logger.info("Cleaning up tuning container...")
        cmd = docker_stop_cmd(TUNE_CONTAINER_NAME)
        run_command_on_host(
            self.host, cmd,
            ssh_kwargs=self.ssh_kwargs, timeout=30, dry_run=self.dry_run,
        )


def build_tuning_command(model: str, tp_size: int, triton_version: str | None = None) -> str:
    """Build the tuning command string for display/testing.

    Args:
        model: Model name.
        tp_size: Tensor parallel size.
        triton_version: Triton version string (e.g. ``"3.6.0"``).  When
            provided the output directory is versioned as
            ``<base>/triton_<version>`` (dots replaced with underscores) so
            configs from different Triton releases don't collide.

    Returns:
        The tuning command string.
    """
    if triton_version:
        config_dir = "%s/triton_%s" % (
            TUNING_CONTAINER_OUTPUT_PATH,
            triton_version.replace(".", "_"),
        )
    else:
        config_dir = TUNING_CONTAINER_OUTPUT_PATH
    return (
        "cd %s && "
        "SGLANG_MOE_CONFIG_DIR=%s "
        "python3 benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py "
        "--model %s --tp-size %d --tune"
    ) % (SGLANG_CLONE_DIR, config_dir, model, tp_size)
