"""CLI integration tests for sparkrun.

Tests the CLI using Click's CliRunner. The CLI is defined in sparkrun.cli
with the main group command.
"""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main


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
        assert "sparkrun, version 0.1.0" in result.output

    def test_help(self, runner):
        """Test that sparkrun --help shows group help text with command names."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sparkrun" in result.output.lower()
        # Check for main commands
        assert "run" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "validate" in result.output
        assert "stop" in result.output

    def test_run_help(self, runner):
        """Test that sparkrun run --help shows run command help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run an inference recipe" in result.output
        assert "--solo" in result.output
        assert "--hosts" in result.output
        assert "--dry-run" in result.output


class TestListCommand:
    """Test the list command."""

    def test_list_shows_bundled_recipes(self, runner):
        """Test that sparkrun list output includes bundled recipe names."""
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        # Check for at least one bundled recipe
        output_lower = result.output.lower()
        assert "glm-4.7-flash-awq" in output_lower or "glm" in output_lower

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

    def test_show_bundled_recipe(self, runner):
        """Test that sparkrun show glm-4.7-flash-awq displays recipe details with VRAM."""
        result = runner.invoke(main, ["show", "glm-4.7-flash-awq"])
        assert result.exit_code == 0
        # Check for recipe detail fields
        assert "Name:" in result.output
        assert "Runtime:" in result.output
        assert "Model:" in result.output
        assert "Container:" in result.output
        # Check for specific recipe values
        assert "GLM-4.7-Flash-AWQ" in result.output or "glm" in result.output.lower()
        assert "vllm" in result.output.lower()
        # VRAM estimation shown by default
        assert "VRAM Estimation" in result.output

    def test_show_nonexistent_recipe(self, runner):
        """Test that sparkrun show nonexistent-recipe exits with error code."""
        result = runner.invoke(main, ["show", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestVramCommand:
    """Test the vram command."""

    def test_vram_bundled_recipe(self, runner):
        """Test sparkrun vram on a bundled recipe shows estimation."""
        result = runner.invoke(main, ["vram", "glm-4.7-flash-awq", "--no-auto-detect"])
        assert result.exit_code == 0
        assert "VRAM Estimation" in result.output
        assert "Model weights:" in result.output
        assert "Per-GPU total:" in result.output
        assert "DGX Spark fit:" in result.output

    def test_vram_with_gpu_mem(self, runner):
        """Test sparkrun vram with --gpu-mem shows budget analysis."""
        result = runner.invoke(main, [
            "vram", "glm-4.7-flash-awq",
            "--no-auto-detect",
            "--gpu-mem", "0.9",
        ])
        assert result.exit_code == 0
        assert "GPU Memory Budget" in result.output
        assert "gpu_memory_utilization" in result.output
        assert "Available for KV" in result.output

    def test_vram_with_tp(self, runner):
        """Test sparkrun vram with --tp override."""
        result = runner.invoke(main, [
            "vram", "glm-4.7-flash-awq",
            "--no-auto-detect",
            "--tp", "2",
        ])
        assert result.exit_code == 0
        assert "Tensor parallel:  2" in result.output

    def test_vram_nonexistent_recipe(self, runner):
        """Test sparkrun vram on nonexistent recipe exits with error."""
        result = runner.invoke(main, ["vram", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_show_no_vram_flag(self, runner):
        """Test sparkrun show --no-vram suppresses VRAM estimation."""
        result = runner.invoke(main, ["show", "glm-4.7-flash-awq", "--no-vram"])
        assert result.exit_code == 0
        assert "VRAM Estimation" not in result.output


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_valid_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun validate glm-4.7-flash-awq exits 0 with 'is valid' message."""
        result = runner.invoke(main, ["validate", "glm-4.7-flash-awq"])
        assert result.exit_code == 0
        assert "is valid" in result.output

    def test_validate_nonexistent_recipe(self, runner, reset_bootstrap):
        """Test that sparkrun validate nonexistent-recipe exits with error."""
        result = runner.invoke(main, ["validate", "nonexistent-recipe"])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestRunCommand:
    """Test the run command (dry-run only)."""

    def test_run_dry_run_solo(self, runner, reset_bootstrap):
        """Test sparkrun run glm-4.7-flash-awq --solo --dry-run --hosts localhost.

        Should show runtime info and exit 0.
        """
        # Mock run_solo to prevent actual SSH execution
        with mock.patch("sparkrun.orchestration.solo.run_solo") as mock_run_solo:
            mock_run_solo.return_value = 0

            result = runner.invoke(main, [
                "run",
                "glm-4.7-flash-awq",
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

            # Verify run_solo was called with dry_run=True
            mock_run_solo.assert_called_once()
            call_kwargs = mock_run_solo.call_args.kwargs
            assert call_kwargs["dry_run"] is True

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

        result = runner.invoke(main, ["stop"])

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
