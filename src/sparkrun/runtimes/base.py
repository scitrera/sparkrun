"""Base class for sparkrun runtimes."""

from __future__ import annotations

import logging
from abc import abstractmethod
from logging import Logger
from typing import Any, TYPE_CHECKING

from scitrera_app_framework import Plugin, Variables, ext_parse_bool

if TYPE_CHECKING:
    from sparkrun.config import SparkrunConfig
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)

EXT_RUNTIME = "sparkrun.runtime"


class RuntimePlugin(Plugin):
    """Abstract base class for sparkrun inference runtimes.

    Each runtime is an SAF Plugin that registers as a multi-extension
    under the 'sparkrun.runtime' extension point. Multiple runtimes
    can coexist simultaneously.

    Subclasses must define:
        - runtime_name: str identifier (e.g. "vllm", "sglang")
        - generate_command(): produce the serve command from a recipe
        - resolve_container(): determine which container image to use
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    runtime_name: str = ""
    default_image_prefix: str = ""

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.runtime.%s" % self.runtime_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_RUNTIME

    def is_enabled(self, v: Variables) -> bool:
        # Must return False for multi-extension plugins to prevent SAF's
        # single-extension cache (er[ext_name]) from short-circuiting
        # subsequent plugin initializations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> RuntimePlugin:
        return self

    # --- Runtime interface ---

    @abstractmethod
    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None) -> str:
        """Generate the serve command string from recipe + CLI overrides.

        Args:
            recipe: The loaded recipe
            overrides: CLI override values (e.g. --port 9000)
            is_cluster: Whether running in multi-node mode
            num_nodes: Total number of nodes in the cluster
            head_ip: Head node IP (only set for cluster mode)

        Returns:
            The full command string to execute inside the container
        """
        ...

    @abstractmethod
    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve the container image to use.

        Args:
            recipe: The loaded recipe
            overrides: Optional CLI overrides

        Returns:
            Fully qualified container image reference
        """
        ...

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return runtime-specific environment variables for cluster mode.

        Override in subclasses to inject runtime-specific cluster config.
        """
        return {}

    def cluster_strategy(self) -> str:
        """Return the clustering strategy for multi-node mode.

        Returns:
            ``"ray"`` — use Ray cluster orchestration (start Ray head/workers,
            then exec serve command on head). This is the default.

            ``"native"`` — the runtime handles its own distribution. Each node
            runs the serve command directly with node-rank arguments appended.
            Used by sglang, which has built-in multi-node support via
            ``--dist-init-addr``, ``--nnodes``, ``--node-rank``.
        """
        return "ray"

    def generate_node_command(
            self,
            recipe: Recipe,
            overrides: dict[str, Any],
            head_ip: str,
            num_nodes: int,
            node_rank: int,
            init_port: int = 25000,
    ) -> str:
        """Generate the serve command for a specific node in native clustering.

        Only called when :meth:`cluster_strategy` returns ``"native"``.

        Args:
            recipe: The loaded recipe.
            overrides: CLI override values.
            head_ip: Head node IP address.
            num_nodes: Total number of nodes.
            node_rank: This node's rank (0 = head).
            init_port: Coordination port for distributed init.

        Returns:
            The full command string for this node.
        """
        raise NotImplementedError(
            "%s does not implement native clustering" % type(self).__name__
        )

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Return list of warnings/errors for runtime-specific fields.

        The base implementation checks that a model is specified.
        Subclasses should call ``super().validate_recipe(recipe)`` and
        extend the returned list with runtime-specific checks.
        """
        issues = []
        if not recipe.model:
            issues.append("[%s] model is required" % self.runtime_name)
        return issues

    @staticmethod
    def build_flags_from_map(
            config,
            flag_map: dict[str, str],
            bool_keys: set[str] | frozenset[str] = frozenset(),
            skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        """Build CLI flag list from a config-key to CLI-flag mapping.

        Iterates *flag_map* and looks up each key in *config*.  Keys in
        *bool_keys* are treated as boolean toggles (flag appended when
        truthy, omitted otherwise).  All other keys emit ``[flag, value]``
        pairs.  Keys listed in *skip_keys* are skipped entirely.

        Args:
            config: Config chain object (must support ``.get(key)``).
            flag_map: Mapping of recipe config key to CLI flag string.
            bool_keys: Set of keys that should be treated as boolean flags.
            skip_keys: Keys to skip (already handled by the caller).

        Returns:
            Flat list of CLI argument strings.
        """
        parts: list[str] = []
        for key, flag in flag_map.items():
            if key in skip_keys:
                continue
            value = config.get(key)
            if value is None:
                continue
            if key in bool_keys:
                if ext_parse_bool(value):
                    parts.append(flag)
            else:
                parts.extend([flag, str(value)])
        return parts

    def is_delegating_runtime(self) -> bool:
        """True if this runtime delegates entirely to external scripts.

        Delegating runtimes (like eugr-vllm) bypass sparkrun's orchestration
        layer and instead call external tools directly.
        """
        return False

    # --- Log following interface ---

    def follow_logs(
            self,
            hosts: list[str],
            cluster_id: str = "sparkrun0",
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            tail: int = 100,
    ) -> None:
        """Follow container logs after a successful launch.

        The default implementation follows the solo container on the
        first host.  Runtimes that use different container naming for
        cluster mode should override this method.

        Args:
            hosts: List of hostnames/IPs (first = head).
            cluster_id: Cluster identifier used when launching.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.
            tail: Number of existing log lines to show before following.
        """
        from sparkrun.orchestration.primitives import build_ssh_kwargs
        from sparkrun.orchestration.docker import generate_container_name
        from sparkrun.orchestration.ssh import stream_remote_logs

        host = hosts[0] if hosts else "localhost"
        container_name = generate_container_name(cluster_id, "solo")
        ssh_kwargs = build_ssh_kwargs(config)

        stream_remote_logs(
            host, container_name, tail=tail, dry_run=dry_run, **ssh_kwargs,
        )

    # --- Launch / Stop interface ---
    #
    # Runtimes control their own orchestration by overriding run() and stop().
    # The default implementations handle solo (single-node) mode.  Runtimes
    # that support multi-node clustering override these to compose their
    # specific flow from orchestration primitives.

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
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            detached: bool = True,
            skip_ib_detect: bool = False,
            nccl_env: dict[str, str] | None = None,
            ib_ip_map: dict[str, str] | None = None,
            **kwargs,
    ) -> int:
        """Launch the workload — solo or cluster.

        The default implementation handles solo (single-node) mode:
        IB detection, container launch, serve command execution.

        Runtimes that support multi-node clustering should override this
        method and compose their flow from orchestration primitives.

        Args:
            hosts: List of hostnames/IPs (first = head).
            image: Container image to use.
            serve_command: The inference serve command to run.
            recipe: The loaded recipe.
            overrides: CLI override values.
            cluster_id: Identifier for container naming.
            env: Additional environment variables from the recipe.
            cache_dir: HuggingFace cache directory path.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.
            detached: Run serve command in background.
            skip_ib_detect: Skip InfiniBand detection.
            nccl_env: Pre-detected NCCL environment variables.  When
                provided (not ``None``), skips runtime IB detection and
                uses this env directly.
            ib_ip_map: Pre-detected InfiniBand IP mapping
                (management host → IB IP).  Used by runtimes that need
                IB addresses for inter-node communication (e.g. llama.cpp
                RPC).  When ``None``, the runtime may detect IB IPs
                itself if ``skip_ib_detect`` is ``False``.
            **kwargs: Runtime-specific keyword arguments.

        Returns:
            Exit code (0 = success).
        """
        return self._run_solo(
            host=hosts[0] if hosts else "localhost",
            image=image,
            serve_command=serve_command,
            cluster_id=cluster_id,
            env=env,
            cache_dir=cache_dir,
            config=config,
            dry_run=dry_run,
            detached=detached,
            skip_ib_detect=skip_ib_detect,
            nccl_env=nccl_env,
        )

    def stop(
            self,
            hosts: list[str],
            cluster_id: str = "sparkrun0",
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> int:
        """Stop a running workload.

        The default implementation handles solo (single-node) teardown.
        Runtimes that support multi-node should override.

        Args:
            hosts: List of hostnames/IPs in the workload.
            cluster_id: Cluster identifier used when launching.
            config: SparkrunConfig instance for SSH settings.
            dry_run: Show what would be done without executing.

        Returns:
            Exit code (0 = success).
        """
        return self._stop_solo(
            host=hosts[0] if hosts else "localhost",
            cluster_id=cluster_id,
            config=config,
            dry_run=dry_run,
        )

    # --- Default solo implementation (used by base and simple runtimes) ---

    def _run_solo(
            self,
            host: str,
            image: str,
            serve_command: str,
            cluster_id: str = "sparkrun0",
            env: dict[str, str] | None = None,
            cache_dir: str | None = None,
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            detached: bool = True,
            skip_ib_detect: bool = False,
            nccl_env: dict[str, str] | None = None,
    ) -> int:
        """Launch a single-node inference workload.

        Steps:
        1. Detect InfiniBand on the target host (optional).
        2. Launch container with ``sleep infinity``.
        3. Execute the serve command inside the container.
        """
        import time
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            merge_env,
            detect_infiniband,
            detect_infiniband_local,
            run_script_on_host,
        )
        from sparkrun.orchestration.docker import generate_container_name
        from sparkrun.orchestration.scripts import (
            generate_container_launch_script,
            generate_exec_serve_script,
        )
        from sparkrun.hosts import is_local_host

        is_local = is_local_host(host)
        container_name = generate_container_name(cluster_id, "solo")
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir)
        all_env = merge_env(env)

        # Step 1: InfiniBand detection (skip if pre-detected nccl_env provided)
        t0 = time.monotonic()
        if nccl_env is not None:
            logger.info("Step 1/3: Using pre-detected NCCL env (%d vars)", len(nccl_env))
        elif not skip_ib_detect:
            nccl_env = {}
            logger.info("Step 1/3: Detecting InfiniBand on %s...", host)
            if is_local:
                nccl_env = detect_infiniband_local(dry_run=dry_run)
            else:
                nccl_env = detect_infiniband(
                    [host], ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                )
            logger.info("Step 1/3: IB detection done (%.1fs)", time.monotonic() - t0)
        else:
            nccl_env = {}
            logger.info("Step 1/3: Skipping InfiniBand detection")

        # Step 2: Launch container
        t0 = time.monotonic()
        logger.info(
            "Step 2/3: Launching container %s on %s (image: %s)...",
            container_name, host, image,
        )
        launch_script = generate_container_launch_script(
            image=image,
            container_name=container_name,
            command="sleep infinity",
            env=all_env,
            volumes=volumes,
            nccl_env=nccl_env,
        )
        result = run_script_on_host(
            host, launch_script, ssh_kwargs=ssh_kwargs, timeout=120, dry_run=dry_run,
        )
        if not result.success and not dry_run:
            logger.error("Failed to launch container: %s", result.stderr)
            return 1
        logger.info("Step 2/3: Container launched (%.1fs)", time.monotonic() - t0)

        # Step 3: Execute serve command
        t0 = time.monotonic()
        logger.info("Step 3/3: Executing serve command in %s...", container_name)
        logger.debug("Serve command: %s", serve_command)
        exec_script = generate_exec_serve_script(
            container_name=container_name,
            serve_command=serve_command,
            env=all_env,
            detached=detached,
        )
        result = run_script_on_host(
            host, exec_script, ssh_kwargs=ssh_kwargs, timeout=60, dry_run=dry_run,
        )
        logger.info("Step 3/3: Serve command dispatched (%.1fs)", time.monotonic() - t0)

        if dry_run:
            return 0
        return result.returncode

    def _stop_solo(
            self,
            host: str,
            cluster_id: str = "sparkrun0",
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> int:
        """Stop a solo workload by removing the container."""
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            cleanup_containers,
            cleanup_containers_local,
        )
        from sparkrun.orchestration.docker import generate_container_name
        from sparkrun.hosts import is_local_host

        container_name = generate_container_name(cluster_id, "solo")
        is_local = is_local_host(host)

        if is_local:
            cleanup_containers_local([container_name], dry_run=dry_run)
        else:
            ssh_kwargs = build_ssh_kwargs(config)
            cleanup_containers([host], [container_name], ssh_kwargs=ssh_kwargs, dry_run=dry_run)

        logger.info("Solo workload '%s' stopped on %s", cluster_id, host)
        return 0

    def __repr__(self) -> str:
        return "%s(runtime_name=%r)" % (self.__class__.__name__, self.runtime_name)
