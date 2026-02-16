"""Native vLLM runtime for sparkrun."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

logger = logging.getLogger(__name__)

# Standard vLLM CLI flags and their recipe default keys
_VLLM_FLAG_MAP = {
    "port": "--port",
    "host": "--host",
    "tensor_parallel": "-tp",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_num_seqs": "--max-num-seqs",
    "served_model_name": "--served-model-name",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "enforce_eager": "--enforce-eager",
    "enable_prefix_caching": "--enable-prefix-caching",
    "trust_remote_code": "--trust-remote-code",
    "distributed_executor_backend": "--distributed-executor-backend",
    "pipeline_parallel": "-pp",
    "kv_cache_dtype": "--kv-cache-dtype",
}

# Boolean flags (present = True, absent = False)
_VLLM_BOOL_FLAGS = {
    "enforce_eager", "enable_prefix_caching", "trust_remote_code",
    "enable_auto_tool_choice",
}


class VllmRuntime(RuntimePlugin):
    """Native vLLM runtime using prebuilt container images."""

    runtime_name = "vllm"
    default_image_prefix = "scitrera/dgx-spark-vllm"

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve the container image to use for vLLM."""
        if recipe.container:
            return recipe.container
        # Fallback: construct from prefix + default tag
        return "%s:latest" % self.default_image_prefix

    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None) -> str:
        """Generate the vllm serve command."""
        config = recipe.build_config_chain(overrides)

        # If recipe has an explicit command template, render it
        rendered = recipe.render_command(config)
        if rendered:
            return rendered

        # Otherwise, build command from structured defaults
        return self._build_command(recipe, config, is_cluster, num_nodes)

    def _build_command(self, recipe: Recipe, config, is_cluster: bool, num_nodes: int) -> str:
        """Build the vllm serve command from structured config."""
        parts = ["vllm", "serve", recipe.model]

        # Auto-inject cluster args
        if is_cluster:
            tp = config.get("tensor_parallel")
            if tp:
                parts.extend(["-tp", str(tp)])
            parts.extend(["--distributed-executor-backend", "ray"])
        else:
            tp = config.get("tensor_parallel")
            if tp:
                parts.extend(["-tp", str(tp)])

        # Add flags from defaults (skip tp since handled above)
        for key, flag in _VLLM_FLAG_MAP.items():
            if key in ("tensor_parallel", "distributed_executor_backend") and is_cluster:
                continue  # already handled
            if key == "tensor_parallel" and not is_cluster:
                continue  # already handled
            value = config.get(key)
            if value is None:
                continue
            if key in _VLLM_BOOL_FLAGS:
                if value and str(value).lower() not in ("false", "0", "no"):
                    parts.append(flag)
            else:
                parts.extend([flag, str(value)])

        return " ".join(parts)

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return vLLM-specific cluster environment variables."""
        return {
            "RAY_memory_monitor_refresh_ms": "0",
        }

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate vLLM-specific recipe fields."""
        issues = []
        if not recipe.model:
            issues.append("[vllm] model is required")
        return issues
