"""Kernel tuning support for sparkrun.

Provides utilities for running Triton fused MoE kernel tuning on DGX Spark
and auto-mounting the resulting configs in inference runs.  Covers both
SGLang and vLLM runtimes.
"""

from __future__ import annotations

from pathlib import Path

from sparkrun.config import DEFAULT_CACHE_DIR

# ---------------------------------------------------------------------------
# SGLang constants
# ---------------------------------------------------------------------------

TUNING_CACHE_SUBDIR = "tuning/sglang"
TUNING_CONTAINER_PATH = "/root/sglang_tuning_configs"
TUNING_CONTAINER_OUTPUT_PATH = "/tuning_output"

# ---------------------------------------------------------------------------
# vLLM constants
# ---------------------------------------------------------------------------

VLLM_TUNING_CACHE_SUBDIR = "tuning/vllm"
VLLM_TUNING_CONTAINER_PATH = "/root/vllm_tuning_configs"
VLLM_TUNING_CONTAINER_OUTPUT_PATH = "/tuning_output"

# ---------------------------------------------------------------------------
# SGLang helpers
# ---------------------------------------------------------------------------


def get_sglang_tuning_dir() -> Path:
    """Return the host-side directory for SGLang tuning configs.

    Default: ``~/.cache/sparkrun/tuning/sglang/``
    """
    return DEFAULT_CACHE_DIR / TUNING_CACHE_SUBDIR


def get_sglang_tuning_volumes() -> dict[str, str] | None:
    """Return volume mapping for tuning configs if they exist.

    Returns:
        Dict mapping host dir to container dir, or ``None`` if no
        tuning configs are available.
    """
    tuning_dir = get_sglang_tuning_dir()
    if tuning_dir.is_dir() and any(tuning_dir.rglob("*.json")):
        return {str(tuning_dir): TUNING_CONTAINER_PATH}
    return None


def get_sglang_tuning_env() -> dict[str, str] | None:
    """Return env vars for tuning configs if they exist.

    Returns:
        Dict with ``SGLANG_MOE_CONFIG_DIR`` set, or ``None`` if no
        tuning configs are available.
    """
    if get_sglang_tuning_volumes() is not None:
        return {"SGLANG_MOE_CONFIG_DIR": TUNING_CONTAINER_PATH}
    return None


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------


def get_vllm_tuning_dir() -> Path:
    """Return the host-side directory for vLLM tuning configs.

    Default: ``~/.cache/sparkrun/tuning/vllm/``
    """
    return DEFAULT_CACHE_DIR / VLLM_TUNING_CACHE_SUBDIR


def get_vllm_tuning_volumes() -> dict[str, str] | None:
    """Return volume mapping for vLLM tuning configs if they exist.

    Returns:
        Dict mapping host dir to container dir, or ``None`` if no
        tuning configs are available.
    """
    tuning_dir = get_vllm_tuning_dir()
    if tuning_dir.is_dir() and any(tuning_dir.rglob("*.json")):
        return {str(tuning_dir): VLLM_TUNING_CONTAINER_PATH}
    return None


def get_vllm_tuning_env() -> dict[str, str] | None:
    """Return env vars for vLLM tuning configs if they exist.

    Returns:
        Dict with ``VLLM_TUNED_CONFIG_FOLDER`` set, or ``None`` if no
        tuning configs are available.
    """
    if get_vllm_tuning_volumes() is not None:
        return {"VLLM_TUNED_CONFIG_FOLDER": VLLM_TUNING_CONTAINER_PATH}
    return None
