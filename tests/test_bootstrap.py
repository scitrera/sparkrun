"""Tests for sparkrun.bootstrap module."""

from __future__ import annotations

import pytest

from sparkrun.bootstrap import (
    init_sparkrun,
    get_variables,
    get_runtime,
    list_runtimes,
)
from sparkrun.runtimes.vllm import VllmRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.eugr_vllm import EugrVllmRuntime


def test_init_sparkrun_returns_variables():
    """Verify that init_sparkrun returns a Variables instance.

    Tests that the initialization function properly returns the SAF Variables object.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")

    assert v is not None
    assert hasattr(v, "runtime_config")  # SAF Variables attribute


def test_init_sparkrun_idempotent():
    """Verify that calling init_sparkrun twice returns the same instance.

    Tests that the module-level singleton pattern works correctly.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v1 = init_sparkrun(log_level="WARNING")
    v2 = init_sparkrun(log_level="WARNING")

    assert v1 is v2


def test_list_runtimes_discovers_all_three():
    """Verify that all three runtimes are discovered: eugr-vllm, sglang, vllm.

    Tests the runtime plugin discovery mechanism.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")
    runtimes = list_runtimes(v=v)

    assert "eugr-vllm" in runtimes
    assert "sglang" in runtimes
    assert "vllm" in runtimes
    assert len(runtimes) == 3


def test_get_runtime_vllm():
    """Get the vllm runtime and verify it's a VllmRuntime with correct name.

    Tests retrieval of the native vLLM runtime plugin.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")
    runtime = get_runtime("vllm", v=v)

    assert isinstance(runtime, VllmRuntime)
    assert runtime.runtime_name == "vllm"


def test_get_runtime_sglang():
    """Get the sglang runtime and verify it's a SglangRuntime.

    Tests retrieval of the SGLang runtime plugin.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")
    runtime = get_runtime("sglang", v=v)

    assert isinstance(runtime, SglangRuntime)
    assert runtime.runtime_name == "sglang"


def test_get_runtime_eugr_vllm():
    """Get the eugr-vllm runtime and verify it's an EugrVllmRuntime.

    Tests retrieval of the EUGR vLLM delegating runtime plugin.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")
    runtime = get_runtime("eugr-vllm", v=v)

    assert isinstance(runtime, EugrVllmRuntime)
    assert runtime.runtime_name == "eugr-vllm"


def test_get_runtime_unknown_raises():
    """Verify that get_runtime raises ValueError for unknown runtime names.

    Tests error handling for invalid runtime requests.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")

    with pytest.raises(ValueError) as exc_info:
        get_runtime("nonexistent-runtime", v=v)

    error_msg = str(exc_info.value).lower()
    assert "unknown runtime" in error_msg or "not found" in error_msg


def test_runtime_classes_are_distinct():
    """Verify that all 3 returned runtime instances are different classes.

    Tests that each runtime plugin is a distinct type.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")

    vllm_rt = get_runtime("vllm", v=v)
    sglang_rt = get_runtime("sglang", v=v)
    eugr_rt = get_runtime("eugr-vllm", v=v)

    assert type(vllm_rt) != type(sglang_rt)
    assert type(vllm_rt) != type(eugr_rt)
    assert type(sglang_rt) != type(eugr_rt)


def test_get_variables_uses_singleton():
    """Verify that get_variables() returns the module singleton.

    Tests that get_variables properly retrieves or initializes the singleton.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    # First call should initialize
    v1 = get_variables()
    assert v1 is not None

    # Second call should return same instance
    v2 = get_variables()
    assert v1 is v2


def test_runtime_is_multi_extension():
    """Verify that runtime plugins are properly configured as multi-extension.

    Tests the SAF multi-extension plugin pattern for runtime plugins.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")
    runtime = get_runtime("vllm", v=v)

    # RuntimePlugin should be multi-extension to allow multiple runtimes
    assert runtime.is_multi_extension(v) is True


def test_runtime_extension_point_name():
    """Verify that all runtimes use the correct extension point.

    Tests that runtime plugins register under 'sparkrun.runtime'.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")

    vllm_rt = get_runtime("vllm", v=v)
    sglang_rt = get_runtime("sglang", v=v)
    eugr_rt = get_runtime("eugr-vllm", v=v)

    from sparkrun.bootstrap import EXT_RUNTIME

    assert vllm_rt.extension_point_name(v) == EXT_RUNTIME
    assert sglang_rt.extension_point_name(v) == EXT_RUNTIME
    assert eugr_rt.extension_point_name(v) == EXT_RUNTIME


def test_list_runtimes_sorted():
    """Verify that list_runtimes returns runtimes in sorted order.

    Tests the alphabetical ordering of runtime names.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    v = init_sparkrun(log_level="WARNING")
    runtimes = list_runtimes(v=v)

    assert runtimes == sorted(runtimes)


def test_init_with_custom_log_level():
    """Test initialization with different log levels.

    Verifies that custom log levels are properly passed through.
    """
    import sparkrun.bootstrap
    sparkrun.bootstrap._variables = None

    # Should not raise an error
    v = init_sparkrun(log_level="DEBUG")
    assert v is not None

    # Reset and try ERROR level
    sparkrun.bootstrap._variables = None
    v = init_sparkrun(log_level="ERROR")
    assert v is not None
