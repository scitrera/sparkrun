"""Shared internals for sparkrun tuning modules.

This is a private module — external code should import from
:mod:`sparkrun.tuning.sglang` or :mod:`sparkrun.tuning.vllm`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.config import DEFAULT_CACHE_DIR

if TYPE_CHECKING:
    from sparkrun.config import SparkrunConfig

logger = logging.getLogger(__name__)

DEFAULT_TP_SIZES = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parameterized host-side helpers
# ---------------------------------------------------------------------------


def _get_tuning_dir(cache_subdir: str) -> Path:
    """Return the host-side directory for tuning configs under *cache_subdir*."""
    return DEFAULT_CACHE_DIR / cache_subdir


def _get_tuning_volumes(
        tuning_dir_fn: callable,
        container_path: str,
) -> dict[str, str] | None:
    """Return volume mapping for tuning configs if they exist.

    Args:
        tuning_dir_fn: Callable returning the host-side tuning :class:`Path`.
        container_path: Mount target inside the container.

    Returns:
        Dict mapping host dir to container dir, or ``None``.
    """
    tuning_dir = tuning_dir_fn()
    if tuning_dir.is_dir() and any(tuning_dir.rglob("*.json")):
        return {str(tuning_dir): container_path}
    return None


def _get_tuning_env(
        volumes_fn: callable,
        env_var: str,
        container_path: str,
) -> dict[str, str] | None:
    """Return env vars for tuning configs if they exist.

    Args:
        volumes_fn: Callable returning the volume dict (or ``None``).
        env_var: Environment variable name to set.
        container_path: Value to assign to the env var.

    Returns:
        Dict with *env_var* set, or ``None``.
    """
    if volumes_fn() is not None:
        return {env_var: container_path}
    return None


# ---------------------------------------------------------------------------
# BaseTuner — shared orchestration skeleton
# ---------------------------------------------------------------------------


class BaseTuner:
    """Shared tuning orchestration logic.

    Subclasses must set the class attributes below and override
    :meth:`_run_tune_for_tp`.
    """

    # --- Class attributes set by subclasses ---
    runtime_label: str          # "SGLang" or "vLLM"
    container_name: str         # e.g. "sparkrun_tune"
    output_path: str            # container-side output mount point
    clone_script: str           # e.g. "sglang_clone_benchmarks.sh"

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
        self.output_dir = output_dir or str(self._default_output_dir())
        self.skip_clone = skip_clone
        self.dry_run = dry_run

        from sparkrun.orchestration.primitives import build_ssh_kwargs
        self.ssh_kwargs = build_ssh_kwargs(config)

    def _default_output_dir(self) -> Path:
        """Return the default host-side output directory.

        Subclasses override to return their ``get_*_tuning_dir()`` result.
        """
        raise NotImplementedError

    # ----- public entry point -----

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
        logger.info("sparkrun %s Kernel Tuner", self.runtime_label)
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

    # ----- orchestration steps -----

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

        # Ensure output directory exists on the remote host (as the SSH user, not root)
        mkdir_script = "#!/bin/bash\nset -uo pipefail\nmkdir -p %s\n" % self.output_dir
        mkdir_result = run_script_on_host(
            self.host, mkdir_script,
            ssh_kwargs=self.ssh_kwargs, timeout=30, dry_run=self.dry_run,
        )
        if not mkdir_result.success and not self.dry_run:
            logger.error(
                "Failed to create output directory %s: %s",
                self.output_dir, mkdir_result.stderr,
            )
            return 1

        volumes = build_volumes(self.cache_dir)
        # Mount tuning output directory
        volumes[self.output_dir] = self.output_path

        launch_script = generate_container_launch_script(
            image=self.image,
            container_name=self.container_name,
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
        """Step 2: Clone benchmark scripts inside the container."""
        import time
        from sparkrun.orchestration.primitives import run_script_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd
        from sparkrun.scripts import read_script

        t0 = time.monotonic()
        logger.info("Step 2/5: Cloning %s benchmark scripts...", self.runtime_label)

        clone_script = read_script(self.clone_script)
        exec_cmd = docker_exec_cmd(self.container_name, clone_script)

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
            self.container_name,
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
        """Step 4 (per-TP): Run the tuning script for a given TP size.

        Subclasses must override this — each runtime builds a different command.
        """
        raise NotImplementedError

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
            "invocations for %s recipes.", self.runtime_label,
        )
        logger.info("=" * 60)

    def _cleanup_container(self) -> None:
        """Step 5: Remove the tuning container."""
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_stop_cmd

        logger.info("Cleaning up tuning container...")
        cmd = docker_stop_cmd(self.container_name)
        run_command_on_host(
            self.host, cmd,
            ssh_kwargs=self.ssh_kwargs, timeout=30, dry_run=self.dry_run,
        )
