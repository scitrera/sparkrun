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
}

_SGLANG_BOOL_FLAGS = {
    "trust_remote_code", "enable_torch_compile", "disable_radix_cache",
}


class SglangRuntime(RuntimePlugin):
    """Native SGLang runtime using prebuilt container images."""

    runtime_name = "sglang"
    default_image_prefix = "scitrera/dgx-spark-sglang"

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve the container image to use for SGLang."""
        if recipe.container:
            return recipe.container
        return "%s:latest" % self.default_image_prefix

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None) -> str:
        """Generate the sglang launch_server command."""
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            return rendered

        return self._build_command(recipe, config, is_cluster, num_nodes, head_ip)

    def _build_command(self, recipe: Recipe, config, is_cluster: bool,
                       num_nodes: int, head_ip: str | None = None) -> str:
        """Build the sglang launch_server command from structured config."""
        parts = ["python3", "-m", "sglang.launch_server", "--model-path", recipe.model]

        # Cluster-specific args
        if is_cluster:
            tp = config.get("tensor_parallel")
            if tp:
                parts.extend(["--tp-size", str(tp)])
            if head_ip:
                parts.extend(["--dist-init-addr", "%s:20000" % head_ip])
                parts.extend(["--nnodes", str(num_nodes)])
        else:
            tp = config.get("tensor_parallel")
            if tp:
                parts.extend(["--tp-size", str(tp)])

        # Add remaining flags
        for key, flag in _SGLANG_FLAG_MAP.items():
            if key == "tensor_parallel":
                continue  # already handled
            value = config.get(key)
            if value is None:
                continue
            if key in _SGLANG_BOOL_FLAGS:
                if value and str(value).lower() not in ("false", "0", "no"):
                    parts.append(flag)
            else:
                parts.extend([flag, str(value)])

        return " ".join(parts)

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return SGLang-specific cluster environment variables."""
        return {
            "NCCL_CUMEM_ENABLE": "0",
        }

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate SGLang-specific recipe fields."""
        issues = []
        if not recipe.model:
            issues.append("[sglang] model is required")
        return issues
