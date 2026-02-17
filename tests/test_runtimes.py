"""Unit tests for sparkrun.runtimes module."""

import re
from unittest import mock

import pytest
from sparkrun.orchestration.docker import generate_cluster_id
from sparkrun.recipe import Recipe
from sparkrun.runtimes.vllm import VllmRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.eugr_vllm import EugrVllmRuntime
from sparkrun.runtimes.base import RuntimePlugin


# --- generate_cluster_id Tests ---

class TestGenerateClusterId:
    """Test deterministic cluster ID generation."""

    def _make_recipe(self, runtime="vllm", model="meta-llama/Llama-2-7b-hf"):
        return Recipe.from_dict({
            "name": "test", "runtime": runtime, "model": model,
        })

    def test_deterministic(self):
        """Same inputs produce the same cluster ID."""
        recipe = self._make_recipe()
        hosts = ["10.0.0.1", "10.0.0.2"]
        assert generate_cluster_id(recipe, hosts) == generate_cluster_id(recipe, hosts)

    def test_host_order_independent(self):
        """Host ordering does not affect the ID (sorted internally)."""
        recipe = self._make_recipe()
        id_a = generate_cluster_id(recipe, ["10.0.0.1", "10.0.0.2"])
        id_b = generate_cluster_id(recipe, ["10.0.0.2", "10.0.0.1"])
        assert id_a == id_b

    def test_different_hosts_differ(self):
        """Different host sets produce different IDs."""
        recipe = self._make_recipe()
        id_a = generate_cluster_id(recipe, ["10.0.0.1"])
        id_b = generate_cluster_id(recipe, ["10.0.0.2"])
        assert id_a != id_b

    def test_prefix_and_format(self):
        """Result starts with 'sparkrun_' followed by 12 hex characters."""
        recipe = self._make_recipe()
        cid = generate_cluster_id(recipe, ["host1"])
        assert re.fullmatch(r"sparkrun_[0-9a-f]{12}", cid)


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
    assert "--dist-init-addr 192.168.1.100:25000" in cmd
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


def test_vllm_cluster_injects_ray_backend_into_template():
    """Cluster mode injects --distributed-executor-backend ray into command templates."""
    recipe_data = {
        "name": "test-recipe",
        "model": "nvidia/some-model",
        "runtime": "vllm",
        "command": "vllm serve {model} -tp {tensor_parallel} --port {port}",
        "defaults": {"tensor_parallel": 2, "port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    # Solo mode: no injection
    cmd_solo = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--distributed-executor-backend" not in cmd_solo

    # Cluster mode: auto-injected
    cmd_cluster = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert "--distributed-executor-backend ray" in cmd_cluster


def test_vllm_cluster_preserves_existing_backend_in_template():
    """Cluster mode does not double-add if template already has the flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "nvidia/some-model",
        "runtime": "vllm",
        "command": "vllm serve {model} --distributed-executor-backend ray",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert cmd.count("--distributed-executor-backend") == 1


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


# --- follow_logs() Tests ---

class _StubRuntime(RuntimePlugin):
    """Minimal concrete runtime for testing base class behaviour."""

    runtime_name = "stub"

    def generate_command(self, recipe, overrides, is_cluster, num_nodes=1, head_ip=None):
        return ""

    def resolve_container(self, recipe, overrides=None):
        return "stub:latest"


class TestBaseFollowLogs:
    """Test base RuntimePlugin.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_solo(self, mock_stream):
        """Base follow_logs calls stream_remote_logs with solo container name."""
        runtime = _StubRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="mytest0",
            config=None,
            dry_run=False,
            tail=50,
        )

        mock_stream.assert_called_once_with(
            "10.0.0.1", "mytest0_solo",
            tail=50, dry_run=False,
        )

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_localhost_default(self, mock_stream):
        """Base follow_logs with empty hosts uses localhost."""
        runtime = _StubRuntime()
        runtime.follow_logs(hosts=[], cluster_id="sparkrun0")

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "localhost"
        assert args[0][1] == "sparkrun0_solo"


class TestVllmFollowLogs:
    """Test VllmRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_tails_serve_log(self, mock_stream):
        """Single-host vllm tails serve log in solo container."""
        runtime = VllmRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == "10.0.0.1"
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_tails_serve_log_on_head(self, mock_stream):
        """Multi-host vllm tails serve log in _head container on hosts[0]."""
        runtime = VllmRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"


class TestSglangFollowLogs:
    """Test SglangRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_uses_file_logs(self, mock_stream):
        """Single-host sglang tails serve log file inside solo container."""
        runtime = SglangRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_cluster_uses_node_0(self, mock_stream):
        """Multi-host sglang follows the _node_0 container on hosts[0]."""
        runtime = SglangRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_node_0"


class TestEugrFollowLogs:
    """Test EugrVllmRuntime.follow_logs() is a no-op."""

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_is_noop(self, mock_stream):
        """eugr-vllm follow_logs does not call stream_remote_logs."""
        runtime = EugrVllmRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_not_called()
