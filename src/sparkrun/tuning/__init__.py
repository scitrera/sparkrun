"""Kernel tuning support for sparkrun.

Provides utilities for running Triton fused MoE kernel tuning on DGX Spark
and auto-mounting the resulting configs in inference runs.  Covers both
SGLang and vLLM runtimes.

SGLang-specific helpers: :mod:`sparkrun.tuning.sglang`
vLLM-specific helpers: :mod:`sparkrun.tuning.vllm`
"""
