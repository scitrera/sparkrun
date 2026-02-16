"""Unit tests for sparkrun.runtimes module."""

import pytest
from sparkrun.recipe import Recipe
from sparkrun.runtimes.vllm import VllmRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.eugr_vllm import EugrVllmRuntime
from sparkrun.runtimes.base import RuntimePlugin


# --- VllmRuntime Tests ---

def test_vllm_runtime_name():
    """VllmRuntime.runtime_name == 'vllm'."""
    runtime = VllmRuntime()
    assert runtime.runtime_name == "vllm"


def test_vllm_resolve_container_from_recipe():
    """Recipe with container field."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "container": "custom-vllm:v1.0",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-vllm:v1.0"


def test_vllm_resolve_container_default():
    """Recipe without container uses default prefix."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-vllm:latest"


def test_vllm_generate_command_from_template():
    """Recipe with command template renders correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "command": "vllm serve {model} --port {port}",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "vllm serve meta-llama/Llama-2-7b-hf --port 8000"


def test_vllm_generate_command_structured():
    """Recipe without template generates vllm serve command from defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "port": 8000,
            "tensor_parallel": 2,
            "gpu_memory_utilization": 0.9,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "-tp 2" in cmd
    assert "--port 8000" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd


def test_vllm_generate_command_cluster():
    """Cluster mode adds --distributed-executor-backend ray."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert "--distributed-executor-backend ray" in cmd
    assert "-tp 4" in cmd


def test_vllm_generate_command_bool_flags():
    """Boolean flags like enforce_eager are handled."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--enforce-eager" in cmd
    # enable_prefix_caching is False, should not appear
    assert "--enable-prefix-caching" not in cmd


def test_vllm_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_vllm_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_vllm_cluster_env():
    """get_cluster_env returns RAY_memory_monitor_refresh_ms."""
    runtime = VllmRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["RAY_memory_monitor_refresh_ms"] == "0"


# --- SglangRuntime Tests ---

def test_sglang_runtime_name():
    """SglangRuntime.runtime_name == 'sglang'."""
    runtime = SglangRuntime()
    assert runtime.runtime_name == "sglang"


def test_sglang_resolve_container():
    """Container resolution."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-sglang:latest"


def test_sglang_generate_command_structured():
    """Generates python3 -m sglang.launch_server with --tp-size, etc."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "defaults": {
            "port": 30000,
            "tensor_parallel": 2,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("python3 -m sglang.launch_server")
    assert "--model-path meta-llama/Llama-2-7b-hf" in cmd
    assert "--tp-size 2" in cmd
    assert "--port 30000" in cmd


def test_sglang_generate_command_cluster():
    """Cluster mode adds --nnodes and --dist-init-addr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(
        recipe, {}, is_cluster=True, num_nodes=2, head_ip="192.168.1.100"
    )
    assert "--dist-init-addr 192.168.1.100:20000" in cmd
    assert "--nnodes 2" in cmd
    assert "--tp-size 4" in cmd


def test_sglang_cluster_env():
    """Returns NCCL_CUMEM_ENABLE."""
    runtime = SglangRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["NCCL_CUMEM_ENABLE"] == "0"


def test_sglang_validate_recipe():
    """Validate recipe."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_sglang_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


# --- EugrVllmRuntime Tests ---

def test_eugr_is_delegating():
    """EugrVllmRuntime.is_delegating_runtime() returns True."""
    runtime = EugrVllmRuntime()
    assert runtime.is_delegating_runtime() is True


def test_eugr_runtime_name():
    """runtime_name == 'eugr-vllm'."""
    runtime = EugrVllmRuntime()
    assert runtime.runtime_name == "eugr-vllm"


def test_eugr_resolve_container():
    """Resolve container for eugr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "container": "custom-eugr:latest",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-eugr:latest"


def test_eugr_resolve_container_default():
    """Default container for eugr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "vllm-node-tf5"


def test_eugr_generate_command():
    """Generate command returns recipe command or empty."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "command": "custom-serve-command",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "custom-serve-command"


def test_eugr_validate_recipe():
    """Validate eugr recipe."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "command": "vllm serve model",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRuntime()

    issues = runtime.validate_recipe(recipe)
    # Should pass validation
    assert all("model is required" not in issue for issue in issues)


# --- Base RuntimePlugin Tests ---

def test_base_runtime_is_enabled_false():
    """RuntimePlugin.is_enabled() returns False (critical for multi-extension)."""
    from scitrera_app_framework.api.variables import Variables

    runtime = RuntimePlugin()
    v = Variables()

    # is_enabled must return False for multi-extension plugins
    assert runtime.is_enabled(v) is False


def test_base_runtime_is_multi_extension_true():
    """RuntimePlugin.is_multi_extension() returns True."""
    from scitrera_app_framework.api.variables import Variables

    runtime = RuntimePlugin()
    v = Variables()

    assert runtime.is_multi_extension(v) is True


def test_base_runtime_is_not_delegating():
    """Base RuntimePlugin.is_delegating_runtime() returns False."""
    runtime = RuntimePlugin()
    assert runtime.is_delegating_runtime() is False


def test_vllm_overrides_in_command():
    """Test that CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    # Override port
    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8000" not in cmd


def test_sglang_overrides_in_command():
    """Test that CLI overrides work for sglang."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "defaults": {"port": 30000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    # Override port
    cmd = runtime.generate_command(recipe, {"port": 31000}, is_cluster=False)
    assert "--port 31000" in cmd
    assert "--port 30000" not in cmd
