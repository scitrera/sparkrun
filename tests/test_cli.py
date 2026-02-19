"""CLI integration tests for sparkrun.

Tests the CLI using Click's CliRunner. The CLI is defined in sparkrun.cli
with the main group command.
"""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.runtimes.sglang import SglangRuntime


@pytest.fixture
def runner():
    """Create a CliRunner instance."""
    return CliRunner()


@pytest.fixture
def reset_bootstrap(v):
    """Ensure sparkrun is initialized before CLI tests that call init_sparkrun().

    By depending on the 'v' fixture, sparkrun is initialized OUTSIDE the
    CliRunner context (where faulthandler.enable() works with real file
    descriptors). The CLI command's init_sparkrun() call then reuses the
    existing singleton instead of re-initializing.
    """
    yield


class TestVersionAndHelp:
    """Test version and help output."""

    def test_version(self, runner):
        """Test that sparkrun --version shows version string."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "sparkrun, version " in result.output

    def test_help(self, runner):
        """Test that sparkrun --help shows group help text with command names."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sparkrun" in result.output.lower()
        # Check for main commands
        assert "run" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "search" in result.output
        assert "stop" in result.output
        assert "logs" in result.output

    def test_run_help(self, runner):
        """Test that sparkrun run --help shows run command help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run an inference recipe" in result.output
        assert "--solo" in result.output
        assert "--hosts" in result.output
        assert "--dry-run" in result.output
        assert "--cluster-id" not in result.output


class TestListCommand:
    """Test the list command."""

    def test_list_shows_recipes(self, runner):
        """Test that sparkrun list discovers recipes from the recipes/ directory."""
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "qwen3-coder-next-fp8-sglang-cluster" in output_lower

    def test_list_table_format(self, runner):
        """Test that list output has header with Name, Runtime, File columns."""
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        # Check for table headers
        assert "Name" in result.output
        assert "Runtime" in result.output
        assert "File" in result.output
        # Check for separator line
        assert "-" * 10 in result.output


class TestShowCommand:
    """Test the show command."""

    def test_show_recipe(self, runner):
        """Test that sparkrun show displays recipe details with VRAM."""
        result = runner.invoke(main, ["show", "qwen3-coder-next-fp8-sglang-cluster"])
        assert result.exit_code == 0
        # Check for recipe detail fields
        assert "Name:" in result.output
        assert "Runtime:" in result.output
        assert "Model:" in result.output
        assert "Container:" in result.output
        # Check for specific recipe values
        assert "qwen3" in result.output.lower()
        assert "sglang" in result.output.lower()
        # VRAM estimation shown by default
        assert "VRAM Estimation" in result.output

    def test_show_nonexistent_recipe(self, runner):
        """Test that sparkrun show nonexistent-recipe exits with error code."""
        result = runner.invoke(main, ["show", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestVramCommand:
    """Test the vram command."""

    def test_vram_recipe(self, runner):
        """Test sparkrun recipe vram shows estimation."""
        result = runner.invoke(main, ["recipe", "vram", "qwen3-coder-next-fp8-sglang-cluster", "--no-auto-detect"])
        assert result.exit_code == 0
        assert "VRAM Estimation" in result.output
        assert "Model weights:" in result.output
        assert "Per-GPU total:" in result.output
        assert "DGX Spark fit:" in result.output

    def test_vram_with_gpu_mem(self, runner):
        """Test sparkrun recipe vram with --gpu-mem shows budget analysis."""
        result = runner.invoke(main, [
            "recipe", "vram", "qwen3-coder-next-fp8-sglang-cluster",
            "--no-auto-detect",
            "--gpu-mem", "0.9",
        ])
        assert result.exit_code == 0
        assert "GPU Memory Budget" in result.output
        assert "gpu_memory_utilization" in result.output
        assert "Available for KV" in result.output

    def test_vram_with_tp(self, runner):
        """Test sparkrun recipe vram with --tp override."""
        result = runner.invoke(main, [
            "recipe", "vram", "qwen3-coder-next-fp8-sglang-cluster",
            "--no-auto-detect",
            "--tp", "4",
        ])
        assert result.exit_code == 0
        assert "Tensor parallel:  4" in result.output

    def test_vram_nonexistent_recipe(self, runner):
        """Test sparkrun recipe vram on nonexistent recipe exits with error."""
        result = runner.invoke(main, ["recipe", "vram", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_show_no_vram_flag(self, runner):
        """Test sparkrun show --no-vram suppresses VRAM estimation."""
        result = runner.invoke(main, ["show", "qwen3-coder-next-fp8-sglang-cluster", "--no-vram"])
        assert result.exit_code == 0
        assert "VRAM Estimation" not in result.output


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_valid_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun recipe validate exits 0 with 'is valid' message."""
        result = runner.invoke(main, ["recipe", "validate", "qwen3-coder-next-fp8-sglang-cluster"])
        assert result.exit_code == 0
        assert "is valid" in result.output

    def test_validate_nonexistent_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun recipe validate nonexistent-recipe exits with error."""
        result = runner.invoke(main, ["recipe", "validate", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestRunCommand:
    """Test the run command (dry-run only)."""

    def test_run_dry_run_solo(self, runner, reset_bootstrap):
        """Test sparkrun run --solo --dry-run --hosts localhost.

        Should show runtime info and exit 0.
        """
        # Mock runtime.run() to prevent actual SSH execution
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts",
                "localhost",
            ])

            assert result.exit_code == 0
            # Check that runtime info is displayed
            assert "Runtime:" in result.output
            assert "Image:" in result.output
            assert "Model:" in result.output
            assert "Mode:" in result.output
            assert "solo" in result.output.lower()

            # Verify runtime.run() was called with dry_run=True
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["dry_run"] is True
            # Cluster ID should be deterministic, not the old default
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["cluster_id"] != "sparkrun0"

    def test_run_nonexistent_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun run nonexistent-recipe --solo --dry-run exits with error."""
        result = runner.invoke(main, [
            "run",
            "nonexistent-recipe",
            "--solo",
            "--dry-run",
        ])

        assert result.exit_code != 0
        assert "Error" in result.output


class TestStopCommand:
    """Test the stop command."""

    def test_stop_no_hosts_error(self, runner, tmp_path, monkeypatch):
        """Test that sparkrun stop with no hosts specified exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "stop",
            "qwen3-coder-next-fp8-sglang-cluster",
        ])

        assert result.exit_code != 0
        # Check error message mentions hosts
        assert "hosts" in result.output.lower() or "Error" in result.output


class TestClusterCommands:
    """Test cluster subcommands: create, list, show, delete, set-default, unset-default, update."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster for CLI tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.cluster_manager import ClusterManager
        mgr = ClusterManager(config_root)
        mgr.create("test-cluster", ["10.0.0.1", "10.0.0.2"])
        return config_root

    def test_cluster_help(self, runner):
        """Test that sparkrun cluster --help shows subcommands."""
        result = runner.invoke(main, ["cluster", "--help"])
        assert result.exit_code == 0
        # Check for cluster subcommands
        assert "create" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "delete" in result.output
        assert "default" in result.output
        assert "set-default" in result.output
        assert "unset-default" in result.output
        assert "update" in result.output

    def test_cluster_create(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "cluster",
            "create",
            "my-cluster",
            "--hosts",
            "host1,host2,host3",
        ])

        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_cluster_create_duplicate(self, runner, cluster_setup):
        """Test that creating a duplicate cluster fails."""
        result = runner.invoke(main, [
            "cluster",
            "create",
            "test-cluster",
            "--hosts",
            "host4,host5",
        ])

        assert result.exit_code != 0
        assert "exists" in result.output.lower() or "Error" in result.output

    def test_cluster_list_empty(self, runner, tmp_path, monkeypatch):
        """Test that cluster list with no clusters shows appropriate message."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["cluster", "list"])

        assert result.exit_code == 0
        assert "No saved clusters" in result.output or "no clusters" in result.output.lower()

    def test_cluster_list_with_clusters(self, runner, cluster_setup):
        """Test that cluster list shows created clusters."""
        result = runner.invoke(main, ["cluster", "list"])

        assert result.exit_code == 0
        assert "test-cluster" in result.output

    def test_cluster_show(self, runner, cluster_setup):
        """Test showing cluster details."""
        result = runner.invoke(main, ["cluster", "show", "test-cluster"])

        assert result.exit_code == 0
        assert "test-cluster" in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_cluster_show_nonexistent(self, runner, cluster_setup):
        """Test that showing a nonexistent cluster fails."""
        result = runner.invoke(main, ["cluster", "show", "nonexistent"])

        assert result.exit_code != 0
        assert "Error" in result.output or "not found" in result.output.lower()

    def test_cluster_delete(self, runner, cluster_setup):
        """Test deleting a cluster with --force flag."""
        result = runner.invoke(main, [
            "cluster",
            "delete",
            "test-cluster",
            "--force",
        ])

        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_cluster_set_default(self, runner, cluster_setup):
        """Test setting a default cluster."""
        result = runner.invoke(main, [
            "cluster",
            "set-default",
            "test-cluster",
        ])

        assert result.exit_code == 0
        assert "Default cluster set" in result.output or "default" in result.output.lower()

    def test_cluster_unset_default(self, runner, cluster_setup):
        """Test unsetting the default cluster."""
        # First set a default
        runner.invoke(main, ["cluster", "set-default", "test-cluster"])

        # Now unset it
        result = runner.invoke(main, ["cluster", "unset-default"])

        assert result.exit_code == 0
        assert "Default cluster unset" in result.output or "unset" in result.output.lower()

    def test_cluster_update(self, runner, cluster_setup):
        """Test updating cluster hosts."""
        result = runner.invoke(main, [
            "cluster",
            "update",
            "test-cluster",
            "--hosts",
            "10.0.0.3,10.0.0.4",
        ])

        assert result.exit_code == 0
        assert "updated" in result.output.lower()

    def test_cluster_create_with_user(self, runner, tmp_path, monkeypatch):
        """Test creating a cluster with --user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "cluster", "create", "my-cluster",
            "--hosts", "host1,host2",
            "--user", "dgxuser",
        ])
        assert result.exit_code == 0

        # Verify user is stored and shown
        result = runner.invoke(main, ["cluster", "show", "my-cluster"])
        assert result.exit_code == 0
        assert "dgxuser" in result.output

    def test_cluster_create_without_user(self, runner, tmp_path, monkeypatch):
        """Test that cluster created without --user does not show User field."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        runner.invoke(main, [
            "cluster", "create", "no-user-cluster",
            "--hosts", "host1,host2",
        ])
        result = runner.invoke(main, ["cluster", "show", "no-user-cluster"])
        assert result.exit_code == 0
        assert "User:" not in result.output

    def test_cluster_update_user(self, runner, cluster_setup):
        """Test updating cluster user."""
        result = runner.invoke(main, [
            "cluster", "update", "test-cluster",
            "--user", "newuser",
        ])
        assert result.exit_code == 0
        assert "updated" in result.output.lower()

        # Verify user is shown
        result = runner.invoke(main, ["cluster", "show", "test-cluster"])
        assert "newuser" in result.output


class TestRunWithCluster:
    """Test run command with --cluster and --hosts-file options."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster for CLI tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.cluster_manager import ClusterManager
        mgr = ClusterManager(config_root)
        mgr.create("test-cluster", ["10.0.0.1", "10.0.0.2"])
        return config_root

    def test_run_help_shows_cluster_option(self, runner):
        """Test that sparkrun run --help shows --cluster and --hosts-file options."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--cluster" in result.output
        assert "--hosts-file" in result.output


class TestTensorParallelValidation:
    """Test tensor_parallel vs host count validation."""

    def test_tp_exceeds_hosts_errors(self, runner, reset_bootstrap):
        """tensor_parallel > number of hosts should exit with error."""
        # qwen3-coder-next-fp8-sglang-cluster has defaults.tensor_parallel=2
        # Provide only 1 host (not --solo) so we hit the validation
        result = runner.invoke(main, [
            "run",
            "qwen3-coder-next-fp8-sglang-cluster",
            "--dry-run",
            "--tp", "4",
            "--hosts", "10.0.0.1,10.0.0.2,10.0.0.3",
        ])

        assert result.exit_code != 0
        assert "tensor_parallel=4" in result.output
        assert "only 3 provided" in result.output

    def test_tp_less_than_hosts_trims(self, runner, reset_bootstrap):
        """tensor_parallel < number of hosts should trim host list."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
            ])

            assert result.exit_code == 0
            assert "tensor_parallel=2" in result.output
            assert "using 2 of 4 hosts" in result.output
            # Should have called with only 2 hosts
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["hosts"]) == 2
            assert call_kwargs["hosts"] == ["10.0.0.1", "10.0.0.2"]

    def test_tp_equals_hosts_uses_all(self, runner, reset_bootstrap):
        """tensor_parallel == number of hosts should use all hosts."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2",
            ])

            assert result.exit_code == 0
            # No trimming message
            assert "using 2 of" not in result.output
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["hosts"]) == 2

    def test_tp_trims_to_one_becomes_solo(self, runner, reset_bootstrap):
        """tensor_parallel=1 with multiple hosts should trim to 1 host and run solo."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--tp", "1",
                "--dry-run",
                "--hosts", "10.0.0.1,10.0.0.2",
            ])

            assert result.exit_code == 0
            assert "tensor_parallel=1" in result.output
            assert "using 1 of 2 hosts" in result.output
            assert "solo" in result.output.lower()
            mock_run.assert_called_once()

    def test_solo_flag_skips_tp_validation(self, runner, reset_bootstrap):
        """--solo flag should skip tensor_parallel validation entirely."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts", "10.0.0.1",
            ])

            assert result.exit_code == 0
            # No trimming or error messages
            assert "tensor_parallel=" not in result.output
            mock_run.assert_called_once()


class TestOptionOverrides:
    """Test --option / -o arbitrary parameter overrides."""

    def test_help_shows_option(self, runner):
        """sparkrun run --help shows --option / -o."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--option" in result.output
        assert "-o" in result.output

    def test_option_overrides_recipe_default(self, runner, reset_bootstrap):
        """--option overrides a recipe default value."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "attention_backend=flashinfer",
            ])

            assert result.exit_code == 0
            # The overrides should contain the option value
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["attention_backend"] == "flashinfer"

    def test_option_multiple(self, runner, reset_bootstrap):
        """Multiple -o flags accumulate."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "attention_backend=triton",
                "-o", "max_model_len=4096",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["overrides"]["attention_backend"] == "triton"
            assert call_kwargs["overrides"]["max_model_len"] == 4096  # auto-coerced to int

    def test_dedicated_cli_param_overrides_option(self, runner, reset_bootstrap):
        """--port takes priority over -o port=XXXX."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "port=9999",
                "--port", "8080",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            # --port should win over -o port=
            assert call_kwargs["overrides"]["port"] == 8080

    def test_option_coerces_types(self, runner, reset_bootstrap):
        """Values are auto-coerced: int, float, bool."""
        with mock.patch.object(SglangRuntime, "run", return_value=0) as mock_run:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
                "-o", "port=8000",
                "-o", "gpu_memory_utilization=0.85",
                "-o", "enforce_eager=true",
                "-o", "served_model_name=my-model",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            ovr = call_kwargs["overrides"]
            assert ovr["port"] == 8000
            assert isinstance(ovr["port"], int)
            assert ovr["gpu_memory_utilization"] == 0.85
            assert isinstance(ovr["gpu_memory_utilization"], float)
            assert ovr["enforce_eager"] is True
            assert ovr["served_model_name"] == "my-model"

    def test_option_bad_format_errors(self, runner, reset_bootstrap):
        """--option without = sign exits with error."""
        result = runner.invoke(main, [
            "run",
            "qwen3-coder-next-fp8-sglang-cluster",
            "--solo",
            "--dry-run",
            "--hosts", "localhost",
            "-o", "bad_no_equals",
        ])

        assert result.exit_code != 0
        assert "must be key=value" in result.output


class TestFollowLogs:
    """Test --no-follow flag and follow_logs integration."""

    def test_run_help_shows_no_follow(self, runner):
        """sparkrun run --help shows --no-follow option."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--no-follow" in result.output

    def test_follow_logs_called_after_successful_run(self, runner, reset_bootstrap):
        """follow_logs is called after a successful detached run."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
             mock.patch.object(SglangRuntime, "run", return_value=0), \
             mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_called_once()
            call_kwargs = mock_follow.call_args.kwargs
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["dry_run"] is False

    def test_no_follow_flag_skips_follow_logs(self, runner, reset_bootstrap):
        """--no-follow prevents follow_logs from being called."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
             mock.patch.object(SglangRuntime, "run", return_value=0), \
             mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--no-follow",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_dry_run_skips_follow_logs(self, runner, reset_bootstrap):
        """--dry-run prevents follow_logs from being called."""
        with mock.patch.object(SglangRuntime, "run", return_value=0), \
             mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--dry-run",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_foreground_skips_follow_logs(self, runner, reset_bootstrap):
        """--foreground prevents follow_logs from being called."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
             mock.patch.object(SglangRuntime, "run", return_value=0), \
             mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--foreground",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 0
            mock_follow.assert_not_called()

    def test_nonzero_exit_skips_follow_logs(self, runner, reset_bootstrap):
        """Non-zero exit code from runtime.run() prevents follow_logs."""
        with mock.patch("sparkrun.orchestration.distribution.distribute_resources", return_value=(None, {}, {})), \
             mock.patch.object(SglangRuntime, "run", return_value=1), \
             mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "run",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--solo",
                "--hosts", "localhost",
            ])

            assert result.exit_code == 1
            mock_follow.assert_not_called()


class TestSetupSshCommand:
    """Test the setup ssh command."""

    @pytest.fixture
    def cluster_setup(self, tmp_path, monkeypatch):
        """Set up a config root with a test cluster for SSH tests."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)
        from sparkrun.cluster_manager import ClusterManager
        mgr = ClusterManager(config_root)
        mgr.create("ssh-cluster", ["10.0.0.1", "10.0.0.2", "10.0.0.3"])
        return config_root

    def test_setup_ssh_help(self, runner):
        """Test that sparkrun setup ssh --help shows relevant options."""
        result = runner.invoke(main, ["setup", "ssh", "--help"])
        assert result.exit_code == 0
        assert "--hosts" in result.output
        assert "--cluster" in result.output
        assert "--user" in result.output
        assert "--dry-run" in result.output
        assert "SSH mesh" in result.output

    def test_setup_ssh_requires_hosts(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, ["setup", "ssh", "--no-include-self"])
        assert result.exit_code != 0
        assert "No hosts" in result.output

    def test_setup_ssh_requires_two_hosts(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh with a single host exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "setup", "ssh", "--hosts", "10.0.0.1", "--no-include-self",
        ])
        assert result.exit_code != 0
        assert "at least 2 hosts" in result.output

    def test_setup_ssh_dry_run(self, runner, tmp_path, monkeypatch):
        """Test that --dry-run shows the command without executing."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--user", "testuser",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Would run:" in result.output
        assert "mesh_ssh_keys.sh" in result.output
        assert "testuser" in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_setup_ssh_dry_run_default_user(self, runner, tmp_path, monkeypatch):
        """Test that --dry-run uses OS user when --user is not specified."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)
        monkeypatch.setenv("USER", "myosuser")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "myosuser" in result.output

    def test_setup_ssh_resolves_cluster(self, runner, cluster_setup):
        """Test that --cluster resolves hosts from a saved cluster."""
        result = runner.invoke(main, [
            "setup", "ssh",
            "--cluster", "ssh-cluster",
            "--user", "ubuntu",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Would run:" in result.output
        assert "ubuntu" in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output
        assert "10.0.0.3" in result.output

    def test_setup_ssh_runs_script(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh invokes subprocess.run with correct args."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as mock_run:
            result = runner.invoke(main, [
                "setup", "ssh",
                "--hosts", "10.0.0.1,10.0.0.2",
                "--user", "testuser",
                "--no-include-self",
            ])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "bash"
            assert "mesh_ssh_keys.sh" in cmd[1]
            assert cmd[2] == "testuser"
            assert cmd[3:] == ["10.0.0.1", "10.0.0.2"]

    def test_setup_ssh_uses_cluster_user(self, runner, tmp_path, monkeypatch):
        """Test that setup ssh picks up the cluster's configured user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.cluster_manager import ClusterManager
        mgr = ClusterManager(config_root)
        mgr.create("usercluster", ["10.0.0.1", "10.0.0.2"], user="dgxuser")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--cluster", "usercluster",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "dgxuser" in result.output

    def test_setup_ssh_cli_user_overrides_cluster_user(self, runner, tmp_path, monkeypatch):
        """Test that --user flag overrides the cluster's configured user."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.cluster_manager import ClusterManager
        mgr = ClusterManager(config_root)
        mgr.create("usercluster2", ["10.0.0.1", "10.0.0.2"], user="dgxuser")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--cluster", "usercluster2",
            "--user", "override_user",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "override_user" in result.output
        # The cluster user should NOT appear in the command
        assert "dgxuser" not in result.output

    def test_setup_ssh_include_self(self, runner, tmp_path, monkeypatch):
        """Test that --include-self adds the local IP to the mesh."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.primitives import local_ip_for
        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--user", "testuser",
            "--include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert local_ip in result.output
        assert "10.0.0.1" in result.output
        assert "10.0.0.2" in result.output

    def test_setup_ssh_include_self_no_duplicate(self, runner, tmp_path, monkeypatch):
        """Test that --include-self doesn't duplicate if local IP already in hosts."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        from sparkrun.orchestration.primitives import local_ip_for
        local_ip = local_ip_for("10.0.0.1")

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,%s" % local_ip,
            "--user", "testuser",
            "--include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        # local IP should appear exactly once in the command line
        cmd_line = result.output.split("Would run:\n")[-1].strip()
        assert cmd_line.count(local_ip) == 1

    def test_setup_ssh_extra_hosts(self, runner, tmp_path, monkeypatch):
        """Test that --extra-hosts adds additional hosts to the mesh."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1",
            "--extra-hosts", "10.0.0.99",
            "--user", "testuser",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "10.0.0.1" in result.output
        assert "10.0.0.99" in result.output

    def test_setup_ssh_extra_hosts_dedup(self, runner, tmp_path, monkeypatch):
        """Test that --extra-hosts deduplicates against --hosts."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "setup", "ssh",
            "--hosts", "10.0.0.1,10.0.0.2",
            "--extra-hosts", "10.0.0.1,10.0.0.3",
            "--user", "testuser",
            "--no-include-self",
            "--dry-run",
        ])
        assert result.exit_code == 0
        cmd_line = result.output.split("Would run:\n")[-1].strip()
        # 10.0.0.1 should appear only once
        assert cmd_line.count("10.0.0.1") == 1
        assert "10.0.0.3" in result.output


class TestLogCommand:
    """Test the logs command."""

    def test_log_help(self, runner):
        """Test that sparkrun logs --help shows relevant options."""
        result = runner.invoke(main, ["logs", "--help"])
        assert result.exit_code == 0
        assert "--tail" in result.output
        assert "--hosts" in result.output
        assert "RECIPE_NAME" in result.output

    def test_log_calls_follow_logs(self, runner, reset_bootstrap):
        """sparkrun logs calls runtime.follow_logs with correct args."""
        with mock.patch.object(SglangRuntime, "follow_logs") as mock_follow:
            result = runner.invoke(main, [
                "logs",
                "qwen3-coder-next-fp8-sglang-cluster",
                "--hosts", "localhost",
                "--tail", "50",
            ])

            assert result.exit_code == 0
            mock_follow.assert_called_once()
            call_kwargs = mock_follow.call_args.kwargs
            assert call_kwargs["cluster_id"].startswith("sparkrun_")
            assert call_kwargs["tail"] == 50

    def test_log_no_hosts_error(self, runner, reset_bootstrap, tmp_path, monkeypatch):
        """sparkrun logs with no hosts exits with error."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        import sparkrun.config
        monkeypatch.setattr(sparkrun.config, "DEFAULT_CONFIG_DIR", config_root)

        result = runner.invoke(main, [
            "logs",
            "qwen3-coder-next-fp8-sglang-cluster",
        ])

        assert result.exit_code != 0
        assert "hosts" in result.output.lower() or "Error" in result.output

    def test_log_nonexistent_recipe(self, runner, reset_bootstrap):
        """sparkrun logs with bad recipe exits with error."""
        result = runner.invoke(main, [
            "logs",
            "nonexistent-recipe",
            "--hosts", "localhost",
        ])

        assert result.exit_code != 0
        assert "Error" in result.output
