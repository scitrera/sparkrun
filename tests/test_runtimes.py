"""Unit tests for sparkrun.runtimes module."""

import re
from unittest import mock

import pytest
from sparkrun.orchestration.job_metadata import generate_cluster_id
from sparkrun.recipe import Recipe
from sparkrun.runtimes.vllm import VllmRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.eugr_vllm import EugrVllmRuntime
from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
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


class TestEugrRunDelegated:
    """Test command construction in EugrVllmRuntime.run_delegated()."""

    @pytest.fixture
    def eugr_recipe(self):
        return Recipe.from_dict({
            "name": "test-recipe",
            "model": "meta-llama/Llama-2-7b-hf",
            "runtime": "eugr-vllm",
            "command": "vllm serve model",
        })

    @pytest.fixture
    def eugr_runtime(self, tmp_path):
        """Create runtime with a fake repo containing run-recipe.sh."""
        runtime = EugrVllmRuntime()
        repo_dir = tmp_path / "eugr-repo"
        repo_dir.mkdir()
        (repo_dir / "run-recipe.sh").write_text("#!/bin/bash\nexit 0\n")
        (repo_dir / "run-recipe.sh").chmod(0o755)
        return runtime, repo_dir

    def test_daemon_flag_passed_when_detached(self, eugr_runtime, eugr_recipe):
        """daemon=True should append --daemon to eugr command."""
        runtime, repo_dir = eugr_runtime
        with mock.patch.object(runtime, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                runtime.run_delegated(eugr_recipe, {}, daemon=True)

                cmd = mock_run.call_args[0][0]
                assert "--daemon" in cmd

    def test_no_daemon_flag_when_foreground(self, eugr_runtime, eugr_recipe):
        """daemon=False (foreground) should not append --daemon."""
        runtime, repo_dir = eugr_runtime
        with mock.patch.object(runtime, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                runtime.run_delegated(eugr_recipe, {}, daemon=False)

                cmd = mock_run.call_args[0][0]
                assert "--daemon" not in cmd

    def test_solo_flag_passed(self, eugr_runtime, eugr_recipe):
        """solo=True should append --solo to eugr command."""
        runtime, repo_dir = eugr_runtime
        with mock.patch.object(runtime, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                runtime.run_delegated(eugr_recipe, {}, solo=True)

                cmd = mock_run.call_args[0][0]
                assert "--solo" in cmd

    def test_hosts_passed_as_comma_list(self, eugr_runtime, eugr_recipe):
        """hosts list should be joined with commas after -n flag."""
        runtime, repo_dir = eugr_runtime
        with mock.patch.object(runtime, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                runtime.run_delegated(
                    eugr_recipe, {}, hosts=["10.0.0.1", "10.0.0.2"],
                )

                cmd = mock_run.call_args[0][0]
                n_idx = cmd.index("-n")
                assert cmd[n_idx + 1] == "10.0.0.1,10.0.0.2"

    def test_run_maps_detached_to_daemon(self, eugr_runtime, eugr_recipe):
        """run() should map detached=True to daemon=True in run_delegated()."""
        runtime, repo_dir = eugr_runtime
        with mock.patch.object(runtime, "run_delegated", return_value=0) as mock_del:
            runtime.run(
                hosts=["10.0.0.1"],
                image="test:latest",
                serve_command="vllm serve",
                recipe=eugr_recipe,
                overrides={},
                detached=True,
            )
            assert mock_del.call_args.kwargs["daemon"] is True

    def test_run_foreground_no_daemon(self, eugr_runtime, eugr_recipe):
        """run() with detached=False should pass daemon=False."""
        runtime, repo_dir = eugr_runtime
        with mock.patch.object(runtime, "run_delegated", return_value=0) as mock_del:
            runtime.run(
                hosts=["10.0.0.1"],
                image="test:latest",
                serve_command="vllm serve",
                recipe=eugr_recipe,
                overrides={},
                detached=False,
            )
            assert mock_del.call_args.kwargs["daemon"] is False


# --- Base RuntimePlugin Tests ---

def test_base_runtime_is_enabled_false():
    """RuntimePlugin.is_enabled() returns False (critical for multi-extension)."""
    from scitrera_app_framework import Variables

    runtime = RuntimePlugin()
    v = Variables()

    # is_enabled must return False for multi-extension plugins
    assert runtime.is_enabled(v) is False


def test_base_runtime_is_multi_extension_true():
    """RuntimePlugin.is_multi_extension() returns True."""
    from scitrera_app_framework import Variables

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

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo(self, mock_stream):
        """Base follow_logs calls stream_container_file_logs with solo container name."""
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

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_localhost_default(self, mock_stream):
        """Base follow_logs with empty hosts uses localhost."""
        runtime = _StubRuntime()
        runtime.follow_logs(hosts=[], cluster_id="sparkrun0")

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "localhost"
        assert args[0][1] == "sparkrun0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_cluster_uses_docker_logs(self, mock_stream):
        """Base _follow_cluster_logs streams docker logs on head node."""
        runtime = _StubRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "test0_head"


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


# --- LlamaCppRuntime Tests ---

def test_llama_cpp_runtime_name():
    """LlamaCppRuntime.runtime_name == 'llama-cpp'."""
    runtime = LlamaCppRuntime()
    assert runtime.runtime_name == "llama-cpp"


def test_llama_cpp_cluster_strategy():
    """LlamaCppRuntime uses native (RPC) clustering, not Ray."""
    runtime = LlamaCppRuntime()
    assert runtime.cluster_strategy() == "native"


def test_llama_cpp_resolve_container_from_recipe():
    """Recipe with container field."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "container": "custom-llama:v1.0",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-llama:v1.0"


def test_llama_cpp_resolve_container_default():
    """Recipe without container uses default prefix."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-llama-cpp:latest"


def test_llama_cpp_generate_command_from_template():
    """Recipe with command template renders correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "command": "llama-server -hf {model} --port {port}",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M --port 8080"


def test_llama_cpp_generate_command_structured_hf():
    """HuggingFace model (contains '/') uses -hf flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "port": 8080,
            "n_gpu_layers": 99,
            "ctx_size": 8192,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd
    assert "--ctx-size 8192" in cmd


def test_llama_cpp_generate_command_gguf_path():
    """Local .gguf path uses -m flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "/models/qwen3-1.7b-q4_k_m.gguf",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("llama-server -m /models/qwen3-1.7b-q4_k_m.gguf")
    assert "--port 8080" in cmd


def test_llama_cpp_generate_command_bool_flags():
    """Boolean flags flash_attn, jinja, no_webui are handled."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "flash_attn": True,
            "jinja": True,
            "no_webui": True,
            "cont_batching": False,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--flash-attn" in cmd
    assert "--jinja" in cmd
    assert "--no-webui" in cmd
    # cont_batching is False, should not appear
    assert "--cont-batching" not in cmd


def test_llama_cpp_generate_command_overrides():
    """CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {"port": 9090}, is_cluster=False)
    assert "--port 9090" in cmd
    assert "--port 8080" not in cmd


def test_llama_cpp_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_llama_cpp_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_llama_cpp_build_rpc_head_command():
    """_build_rpc_head_command appends --rpc with worker addresses."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080, "n_gpu_layers": 99},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()
    config = recipe.build_config_chain({})

    cmd = runtime._build_rpc_head_command(
        recipe, config,
        worker_hosts=["10.0.0.2", "10.0.0.3"],
        rpc_port=50052,
    )
    assert "--rpc 10.0.0.2:50052,10.0.0.3:50052" in cmd
    assert cmd.startswith("llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M")


def test_llama_cpp_build_rpc_worker_command():
    """_build_rpc_worker_command returns rpc-server with host and port."""
    cmd = LlamaCppRuntime._build_rpc_worker_command(50052)
    assert cmd == "rpc-server --host 0.0.0.0 --port 50052"


def test_llama_cpp_container_name():
    """_container_name returns {cluster_id}_{role}."""
    assert LlamaCppRuntime._container_name("spark0", "head") == "spark0_head"
    assert LlamaCppRuntime._container_name("spark0", "worker") == "spark0_worker"


def test_llama_cpp_generate_command_gguf_presync_template():
    """When _gguf_model_path is set, template renders with resolved path
    and -hf is switched to -m."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "port": 8080,
            "host": "0.0.0.0",
            "n_gpu_layers": 99,
            "ctx_size": 8192,
        },
        "command": (
            "llama-server \\\n"
            "    -hf {model} \\\n"
            "    --host {host} \\\n"
            "    --port {port} \\\n"
            "    --n-gpu-layers {n_gpu_layers} \\\n"
            "    --ctx-size {ctx_size} \\\n"
            "    --flash-attn \\\n"
            "    --jinja \\\n"
            "    --no-webui"
        ),
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    gguf_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/snapshots/abc123/q4_k_m.gguf"
    cmd = runtime.generate_command(
        recipe,
        {"_gguf_model_path": gguf_path, "model": gguf_path},
        is_cluster=False,
    )
    # Template is respected: -hf switched to -m, path substituted
    assert "-m " + gguf_path in cmd
    assert "-hf " not in cmd
    # Other flags from template are preserved
    assert "--host 0.0.0.0" in cmd
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd
    assert "--ctx-size 8192" in cmd
    assert "--flash-attn" in cmd
    assert "--jinja" in cmd
    assert "--no-webui" in cmd


def test_llama_cpp_generate_command_gguf_presync_structured():
    """When _gguf_model_path is set and no template, structured build uses -m."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080, "n_gpu_layers": 99},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    gguf_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/snapshots/abc123/q4_k_m.gguf"
    cmd = runtime.generate_command(
        recipe,
        {"_gguf_model_path": gguf_path, "model": gguf_path},
        is_cluster=False,
    )
    assert cmd.startswith("llama-server -m " + gguf_path)
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd


def test_llama_cpp_generate_command_no_presync_uses_hf():
    """Without _gguf_model_path, template renders -hf with original model."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
        "command": "llama-server -hf {model} --port {port}",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "-hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M" in cmd
    assert "--port 8080" in cmd


class TestLlamaCppFollowLogs:
    """Test LlamaCppRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_uses_file_logs(self, mock_stream):
        """Single-host llama-cpp tails serve log file inside solo container."""
        runtime = LlamaCppRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_cluster_uses_docker_logs_on_head(self, mock_stream):
        """Multi-host llama-cpp follows docker logs on _head container."""
        runtime = LlamaCppRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"
