"""Native SGLang runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)

# SGLang CLI flag mapping
_SGLANG_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "--tp-size",
    "gpu_memory_utilization": "--mem-fraction-static",
    "max_model_len": "--context-length",
    "max_num_seqs": "--max-running-requests",
    "served_model_name": "--served-model-name",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "trust_remote_code": "--trust-remote-code",
    "chunked_prefill": "--chunked-prefill-size",
    "kv_cache_dtype": "--kv-cache-dtype",
    "tokenizer_path": "--tokenizer-path",
}

_SGLANG_BOOL_FLAGS = {
    "trust_remote_code", "enable_torch_compile", "disable_radix_cache",
}


class SglangRuntime(RuntimePlugin):
    """Native SGLang runtime using prebuilt container images.

    SGLang uses its own distributed init mechanism for multi-node inference,
    not Ray.  Each node runs the full ``sglang.launch_server`` command with
    ``--dist-init-addr``, ``--nnodes``, and ``--node-rank`` arguments.
    """

    runtime_name = "sglang"
    default_image_prefix = "scitrera/dgx-spark-sglang"

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve the container image to use for SGLang."""
        if recipe.container:
            return recipe.container
        return "%s:latest" % self.default_image_prefix

    def cluster_strategy(self) -> str:
        """SGLang uses native multi-node distribution, not Ray."""
        return "native"

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None) -> str:
        """Generate the sglang launch_server command.

        For cluster mode this produces the *base* command without
        ``--node-rank``.  Use :meth:`generate_node_command` to get the
        per-node variant.
        """
        config = recipe.build_config_chain(overrides)
        self._inject_gguf_model(config)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            return rendered

        return self._build_command(recipe, config, is_cluster, num_nodes, head_ip)

    def generate_node_command(
            self,
            recipe: Recipe,
            overrides: dict[str, Any],
            head_ip: str,
            num_nodes: int,
            node_rank: int,
            init_port: int = 25000,
    ) -> str:
        """Generate the sglang command for a specific node.

        Produces the full ``sglang.launch_server`` invocation with the
        node-specific ``--dist-init-addr``, ``--nnodes``, and
        ``--node-rank`` flags appended.
        """
        config = recipe.build_config_chain(overrides)
        self._inject_gguf_model(config)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            base = rendered
        else:
            base = self._build_base_command(recipe, config)

        # Append sglang multi-node arguments
        parts = [
            base,
            "--dist-init-addr %s:%d" % (head_ip, init_port),
            "--nnodes %d" % num_nodes,
            "--node-rank %d" % node_rank,
        ]
        return " ".join(parts)

    @staticmethod
    def _inject_gguf_model(config) -> None:
        """Ensure ``{model}`` in command templates resolves to the GGUF file path.

        When a GGUF model has been pre-synced, the CLI stores the
        container-internal path as ``_gguf_model_path`` in overrides.
        This helper copies that value into the ``model`` key so that
        ``{model}`` in recipe command templates renders the local file
        path instead of the raw HF repo spec (which includes the
        sparkrun-specific ``:quant`` suffix that runtimes cannot parse).
        """
        gguf_path = config.get("_gguf_model_path")
        if gguf_path:
            config.put("model", str(gguf_path))

    def _build_base_command(self, recipe: Recipe, config) -> str:
        """Build the sglang command without cluster-specific arguments."""
        # For GGUF models, use the resolved file path instead of the HF repo name
        model_path = config.get("_gguf_model_path") or recipe.model
        parts = ["python3", "-m", "sglang.launch_server", "--model-path", str(model_path)]

        tp = config.get("tensor_parallel")
        if tp:
            parts.extend(["--tp-size", str(tp)])

        parts.extend(self.build_flags_from_map(
            config, _SGLANG_FLAG_MAP, bool_keys=_SGLANG_BOOL_FLAGS,
            skip_keys={"tensor_parallel"},
        ))

        return " ".join(parts)

    def _build_command(self, recipe: Recipe, config, is_cluster: bool,
                       num_nodes: int, head_ip: str | None = None) -> str:
        """Build the sglang launch_server command from structured config.

        For cluster mode, includes ``--dist-init-addr`` and ``--nnodes`` but
        NOT ``--node-rank`` (that is added per-node by the orchestrator or
        by :meth:`generate_node_command`).
        """
        base = self._build_base_command(recipe, config)

        if is_cluster and head_ip:
            base += " --dist-init-addr %s:25000 --nnodes %d" % (head_ip, num_nodes)

        return base

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return SGLang-specific cluster environment variables."""
        return {
            "NCCL_CUMEM_ENABLE": "0",
        }

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate SGLang-specific recipe fields."""
        from sparkrun.models.download import is_gguf_model

        issues = super().validate_recipe(recipe)

        if recipe.model and is_gguf_model(recipe.model):
            tokenizer = (recipe.defaults or {}).get("tokenizer_path")
            cmd = recipe.command or ""
            cmd_has_tokenizer = "--tokenizer-path" in cmd or "{tokenizer_path}" in cmd

            if not tokenizer and not cmd_has_tokenizer:
                issues.append(
                    "[sglang] GGUF model detected but no tokenizer path configured. "
                    "SGLang requires --tokenizer-path pointing to the base (non-GGUF) HF model. "
                    "Set 'tokenizer_path' in defaults (e.g. tokenizer_path: Qwen/Qwen3-1.7B) "
                    "or add --tokenizer-path to the command template."
                )
            if tokenizer and cmd and not cmd_has_tokenizer:
                issues.append(
                    "[sglang] GGUF recipe has 'tokenizer_path' in defaults but the command "
                    "template does not reference {tokenizer_path} or --tokenizer-path. "
                    "Add '--tokenizer-path {tokenizer_path}' to the command template."
                )

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
        """Follow serve logs — solo or native cluster head (node_0).

        Solo mode uses ``_run_solo`` which launches ``sleep infinity`` and
        exec's the serve command with output to ``/tmp/sparkrun_serve.log``,
        so we tail that file inside the container (same as vLLM).

        Cluster mode runs the serve command directly as the container
        entrypoint, so ``docker logs`` is correct.
        """
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        if len(hosts) <= 1:
            from sparkrun.orchestration.docker import generate_container_name
            from sparkrun.orchestration.ssh import stream_container_file_logs

            host = hosts[0] if hosts else "localhost"
            container_name = generate_container_name(cluster_id, "solo")
            ssh_kwargs = build_ssh_kwargs(config)

            stream_container_file_logs(
                host, container_name, tail=tail, dry_run=dry_run, **ssh_kwargs,
            )
            return

        from sparkrun.orchestration.ssh import stream_remote_logs

        head_host = hosts[0]
        container_name = self._container_name(cluster_id, 0)
        ssh_kwargs = build_ssh_kwargs(config)

        stream_remote_logs(
            head_host, container_name, tail=tail, dry_run=dry_run, **ssh_kwargs,
        )

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
            nccl_env: dict[str, str] | None = None,
            init_port: int = 25000,
            **kwargs,
    ) -> int:
        """Launch an SGLang workload — solo or native cluster.

        For a single host, delegates to the base solo implementation.
        For multiple hosts, orchestrates a native SGLang cluster where
        each node runs the full serve command with per-node rank args.
        """
        if len(hosts) <= 1:
            return self._run_solo(
                host=hosts[0] if hosts else "localhost",
                image=image, serve_command=serve_command,
                cluster_id=cluster_id, env=env, cache_dir=cache_dir,
                config=config, dry_run=dry_run, detached=detached,
                skip_ib_detect=skip_ib_detect, nccl_env=nccl_env,
            )

        return self._run_native_cluster(
            hosts=hosts, image=image, recipe=recipe, overrides=overrides,
            cluster_id=cluster_id, init_port=init_port, env=env,
            cache_dir=cache_dir, config=config, dry_run=dry_run,
            skip_ib_detect=skip_ib_detect, nccl_env=nccl_env,
        )

    def stop(
            self,
            hosts: list[str],
            cluster_id: str = "sparkrun0",
            config=None,
            dry_run: bool = False,
    ) -> int:
        """Stop an SGLang workload — solo or native cluster."""
        if len(hosts) <= 1:
            return self._stop_solo(
                host=hosts[0] if hosts else "localhost",
                cluster_id=cluster_id, config=config, dry_run=dry_run,
            )

        from sparkrun.orchestration.primitives import build_ssh_kwargs
        from sparkrun.orchestration.ssh import run_remote_command
        from sparkrun.orchestration.docker import docker_stop_cmd

        ssh_kwargs = build_ssh_kwargs(config)
        for rank, host in enumerate(hosts):
            container_name = self._container_name(cluster_id, rank)
            run_remote_command(
                host, docker_stop_cmd(container_name),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )

        logger.info("SGLang cluster '%s' stopped on %d host(s)", cluster_id, len(hosts))
        return 0

    @staticmethod
    def _container_name(cluster_id: str, rank: int) -> str:
        """Generate container name: ``{cluster_id}_node_{rank}``."""
        return "%s_node_%d" % (cluster_id, rank)

    def _run_native_cluster(
            self,
            hosts: list[str],
            image: str,
            recipe: Recipe,
            overrides: dict[str, Any],
            cluster_id: str,
            init_port: int,
            env: dict[str, str] | None,
            cache_dir: str | None,
            config,
            dry_run: bool,
            skip_ib_detect: bool,
            nccl_env: dict[str, str] | None = None,
    ) -> int:
        """Orchestrate a multi-node SGLang cluster using native distribution.

        Steps:
        1. Clean up existing containers on all hosts.
        2. Detect InfiniBand on all hosts (parallel).
        3. Detect head node IP.
        4. Launch head node (rank 0).
        5. Wait for head init port to be ready.
        6. Launch worker nodes in parallel.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sparkrun.orchestration.primitives import (
            build_ssh_kwargs,
            build_volumes,
            merge_env,
            detect_infiniband,
            detect_host_ip,
            wait_for_port,
        )
        from sparkrun.orchestration.ssh import run_remote_script, run_remote_command
        from sparkrun.orchestration.docker import docker_run_cmd, docker_stop_cmd

        num_nodes = len(hosts)
        head_host = hosts[0]
        worker_hosts = hosts[1:]
        ssh_kwargs = build_ssh_kwargs(config)
        volumes = build_volumes(cache_dir)
        runtime_env = self.get_cluster_env(head_ip="<pending>", num_nodes=num_nodes)
        all_env = merge_env(env, runtime_env)

        self._print_native_banner(hosts, image, cluster_id, init_port, dry_run)

        # Step 1: Cleanup
        t0 = time.monotonic()
        logger.info("Step 1/6: Cleaning up existing containers for cluster '%s'...", cluster_id)
        for rank, host in enumerate(hosts):
            container_name = self._container_name(cluster_id, rank)
            run_remote_command(
                host, docker_stop_cmd(container_name),
                timeout=30, dry_run=dry_run, **ssh_kwargs,
            )
        logger.info("Step 1/6: Cleanup done (%.1fs)", time.monotonic() - t0)

        # Step 2: InfiniBand detection (skip if pre-detected nccl_env provided)
        t0 = time.monotonic()
        if nccl_env is not None:
            logger.info("Step 2/6: Using pre-detected NCCL env (%d vars)", len(nccl_env))
        elif not skip_ib_detect:
            nccl_env = {}
            logger.info("Step 2/6: Detecting InfiniBand on all hosts...")
            nccl_env = detect_infiniband(
                hosts, head_host=head_host,
                ssh_kwargs=ssh_kwargs, dry_run=dry_run,
            )
            logger.info("Step 2/6: IB detection done (%.1fs)", time.monotonic() - t0)
        else:
            nccl_env = {}
            logger.info("Step 2/6: Skipping InfiniBand detection")

        # Step 3: Detect head node IP
        t0 = time.monotonic()
        logger.info("Step 3/6: Detecting head node IP on %s...", head_host)
        try:
            head_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        except RuntimeError as e:
            logger.error("%s", e)
            return 1
        logger.info("  Head IP: %s", head_ip)
        logger.info("Step 3/6: IP detection done (%.1fs)", time.monotonic() - t0)

        # Generate per-node commands
        head_command = self.generate_node_command(
            recipe=recipe, overrides=overrides,
            head_ip=head_ip, num_nodes=num_nodes,
            node_rank=0, init_port=init_port,
        )
        logger.info("Serve command (head, rank 0):")
        for line in head_command.strip().splitlines():
            logger.info("  %s", line)

        # Step 4: Launch head node (rank 0)
        t0 = time.monotonic()
        head_container = self._container_name(cluster_id, 0)
        logger.info(
            "Step 4/6: Launching head node (rank 0) on %s as %s...",
            head_host, head_container,
        )
        head_script = self._generate_node_script(
            image=image, container_name=head_container,
            serve_command=head_command, env=all_env,
            volumes=volumes, nccl_env=nccl_env,
        )
        head_result = run_remote_script(
            head_host, head_script, timeout=120, dry_run=dry_run, **ssh_kwargs,
        )
        if not head_result.success and not dry_run:
            logger.error("Failed to launch head node: %s", head_result.stderr[:200])
            return 1
        logger.info("Step 4/6: Head node launched (%.1fs)", time.monotonic() - t0)

        # Step 5: Wait for head init port
        t0 = time.monotonic()
        if not dry_run:
            logger.info("Step 5/6: Waiting for head node init port %s:%d...", head_host, init_port)
            ready = wait_for_port(
                head_host, init_port,
                max_retries=60, retry_interval=2,
                ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                container_name=head_container,
            )
            if not ready:
                logger.error(
                    "Head node failed to become ready. "
                    "Check logs: ssh %s 'docker logs %s'", head_host, head_container,
                )
                return 1
            logger.info("Step 5/6: Head node ready (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 5/6: [dry-run] Would wait for head init port %d", init_port)

        # Step 6: Launch worker nodes in parallel
        t0 = time.monotonic()
        if worker_hosts:
            logger.info(
                "Step 6/6: Launching %d worker node(s) on %s...",
                len(worker_hosts), ", ".join(worker_hosts),
            )
            with ThreadPoolExecutor(max_workers=len(worker_hosts)) as executor:
                futures = {}
                for i, host in enumerate(worker_hosts):
                    rank = i + 1
                    worker_command = self.generate_node_command(
                        recipe=recipe, overrides=overrides,
                        head_ip=head_ip, num_nodes=num_nodes,
                        node_rank=rank, init_port=init_port,
                    )
                    worker_container = self._container_name(cluster_id, rank)
                    worker_script = self._generate_node_script(
                        image=image, container_name=worker_container,
                        serve_command=worker_command, env=all_env,
                        volumes=volumes, nccl_env=nccl_env,
                    )
                    future = executor.submit(
                        run_remote_script, host, worker_script,
                        timeout=120, dry_run=dry_run, **ssh_kwargs,
                    )
                    futures[future] = (host, rank)

                for future in as_completed(futures):
                    host, rank = futures[future]
                    result = future.result()
                    if not result.success and not dry_run:
                        logger.warning(
                            "  Worker rank %d on %s may have failed: %s",
                            rank, host, result.stderr[:100],
                        )

            logger.info("Step 6/6: Workers launched (%.1fs)", time.monotonic() - t0)
        else:
            logger.info("Step 6/6: No worker hosts, skipping")

        self._print_native_connection_info(hosts, cluster_id, head_host, head_ip, init_port)
        return 0

    @staticmethod
    def _generate_node_script(
            image: str,
            container_name: str,
            serve_command: str,
            env: dict[str, str] | None = None,
            volumes: dict[str, str] | None = None,
            nccl_env: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that launches an sglang node directly.

        Unlike the Ray approach, each sglang node runs the full serve
        command as the container's entrypoint.
        """
        from sparkrun.orchestration.docker import docker_run_cmd, docker_stop_cmd
        from sparkrun.orchestration.primitives import merge_env

        all_env = merge_env(nccl_env, env)
        cleanup = docker_stop_cmd(container_name)
        run_cmd = docker_run_cmd(
            image=image,
            command=serve_command,
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
        )

        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            "\n"
            "echo 'Cleaning up existing container: %(name)s'\n"
            "%(cleanup)s\n"
            "\n"
            "echo 'Launching sglang node: %(name)s'\n"
            "%(run_cmd)s\n"
            "\n"
            "# Verify container started\n"
            "sleep 1\n"
            "if docker ps --format '{{.Names}}' | grep -q '^%(name)s$'; then\n"
            "    echo 'Container %(name)s launched successfully'\n"
            "else\n"
            "    echo 'ERROR: Container %(name)s failed to start' >&2\n"
            "    docker logs %(name)s 2>&1 | tail -20 || true\n"
            "    exit 1\n"
            "fi\n"
        ) % {"name": container_name, "cleanup": cleanup, "run_cmd": run_cmd}

    def _print_native_banner(self, hosts, image, cluster_id, init_port, dry_run):
        mode = "DRY-RUN" if dry_run else "LIVE"
        logger.info("=" * 60)
        logger.info("sparkrun SGLang Cluster Launcher")
        logger.info("=" * 60)
        logger.info("Cluster ID:     %s", cluster_id)
        logger.info("Image:          %s", image)
        logger.info("Nodes (%d):     %s", len(hosts), ", ".join(hosts))
        logger.info("Head Node:      %s", hosts[0])
        logger.info("Init Port:      %d", init_port)
        logger.info("Mode:           %s", mode)
        logger.info("=" * 60)

    def _print_native_connection_info(self, hosts, cluster_id, head_host, head_ip, init_port):
        logger.info("=" * 60)
        logger.info("SGLang cluster launched successfully.")
        logger.info("Nodes: %d", len(hosts))
        logger.info("")
        logger.info("To view head logs:")
        logger.info("  sparkrun logs <recipe> --hosts %s", ",".join(hosts))
        logger.info("")
        logger.info("To stop cluster:")
        logger.info("  sparkrun stop <recipe> --hosts %s", ",".join(hosts))
        logger.info("")
        for rank, host in enumerate(hosts):
            logger.info(
                "  Node %d: ssh %s 'docker logs %s'",
                rank, host, self._container_name(cluster_id, rank),
            )
        logger.info("=" * 60)
