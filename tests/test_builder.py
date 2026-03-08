"""Tests for sparkrun builder plugin system."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.core.recipe import Recipe


# ---------------------------------------------------------------------------
# BuilderPlugin base class
# ---------------------------------------------------------------------------

class TestBuilderPluginBase:
    """Tests for BuilderPlugin abstract base class."""

    def _make_plugin(self):
        """Return an instance of the concrete base class."""
        from sparkrun.builders.base import BuilderPlugin
        return BuilderPlugin()

    def test_builder_plugin_is_enabled_false(self):
        """is_enabled returns False — required for multi-extension registration."""
        from scitrera_app_framework import Variables
        plugin = self._make_plugin()
        v = Variables()
        assert plugin.is_enabled(v) is False

    def test_builder_plugin_is_multi_extension_true(self):
        """is_multi_extension returns True."""
        from scitrera_app_framework import Variables
        plugin = self._make_plugin()
        v = Variables()
        assert plugin.is_multi_extension(v) is True

    def test_builder_plugin_prepare_image_returns_unchanged(self):
        """Default prepare_image returns the image arg unchanged."""
        plugin = self._make_plugin()
        recipe = Recipe.from_dict({"name": "test", "model": "some/model", "runtime": "vllm"})
        result = plugin.prepare_image("my-image:latest", recipe, ["10.0.0.1"])
        assert result == "my-image:latest"

    def test_builder_plugin_validate_recipe_returns_empty(self):
        """Default validate_recipe returns an empty list."""
        plugin = self._make_plugin()
        recipe = Recipe.from_dict({"name": "test", "model": "some/model", "runtime": "vllm"})
        assert plugin.validate_recipe(recipe) == []

    def test_builder_plugin_repr(self):
        """repr includes builder_name."""
        from sparkrun.builders.base import BuilderPlugin
        plugin = BuilderPlugin()
        plugin.builder_name = "my-builder"
        r = repr(plugin)
        assert "BuilderPlugin" in r
        assert "my-builder" in r


# ---------------------------------------------------------------------------
# DockerPullBuilder
# ---------------------------------------------------------------------------

class TestDockerPullBuilder:
    """Tests for DockerPullBuilder."""

    def _make_builder(self):
        from sparkrun.builders.docker_pull import DockerPullBuilder
        return DockerPullBuilder()

    def test_docker_pull_builder_name(self):
        """builder_name == 'docker-pull'."""
        assert self._make_builder().builder_name == "docker-pull"

    def test_docker_pull_prepare_image_noop(self):
        """prepare_image returns image unchanged (no subprocess calls)."""
        builder = self._make_builder()
        recipe = Recipe.from_dict({"name": "test", "model": "some/model", "runtime": "vllm"})
        with mock.patch("subprocess.run") as mock_run:
            result = builder.prepare_image("docker-image:latest", recipe, ["10.0.0.1"])
        assert result == "docker-image:latest"
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# EugrBuilder
# ---------------------------------------------------------------------------

@pytest.fixture
def eugr_builder_with_repo(tmp_path):
    """Create EugrBuilder with a fake repo containing build-and-copy.sh and a sample mod."""
    from sparkrun.builders.eugr import EugrBuilder
    builder = EugrBuilder()
    repo_dir = tmp_path / "eugr-repo"
    repo_dir.mkdir()
    build_script = repo_dir / "build-and-copy.sh"
    build_script.write_text("#!/bin/bash\nexit 0\n")
    build_script.chmod(0o755)
    mod_dir = repo_dir / "mods" / "flash-attn"
    mod_dir.mkdir(parents=True)
    (mod_dir / "run.sh").write_text("#!/bin/bash\necho applied\n")
    return builder, repo_dir


class TestEugrBuilderName:
    """Test EugrBuilder identity."""

    def test_eugr_builder_name(self):
        """builder_name == 'eugr'."""
        from sparkrun.builders.eugr import EugrBuilder
        assert EugrBuilder().builder_name == "eugr"


class TestEugrPrepareImage:
    """Test EugrBuilder.prepare_image() — build and mod injection."""

    def test_eugr_prepare_with_build_args(self, eugr_builder_with_repo):
        """prepare_image() calls build-and-copy.sh when build_args are present."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
            "container": "my-image",
            "runtime_config": {"build_args": ["--some-flag"]},
        })
        with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                result = builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        assert result == "my-image"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert str(repo_dir / "build-and-copy.sh") in cmd[0]
        assert "-t" in cmd
        assert "my-image" in cmd
        assert "--some-flag" in cmd

    def test_eugr_prepare_without_build_args_image_exists(self, eugr_builder_with_repo):
        """prepare_image() is a no-op when no build_args/mods and image exists locally."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
        })
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo") as mock_ensure:
                result = builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])
                mock_ensure.assert_not_called()

        assert result == "vllm-node"

    def test_eugr_prepare_builds_when_image_missing(self, eugr_builder_with_repo):
        """prepare_image() triggers a build when image is not found locally."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
            "container": "my-image",
        })
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(returncode=0)
                    result = builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert str(repo_dir / "build-and-copy.sh") in cmd[0]
        assert "-t" in cmd
        assert "my-image" in cmd
        assert result == "my-image"

    def test_eugr_prepare_dry_run(self, eugr_builder_with_repo):
        """prepare_image() in dry-run does not execute subprocess."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
            "runtime_config": {"build_args": ["--flag"]},
        })
        with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                builder.prepare_image("vllm-node", recipe, ["10.0.0.1"], dry_run=True)

        mock_run.assert_not_called()

    def test_eugr_prepare_injects_mod_pre_exec(self, eugr_builder_with_repo):
        """prepare_image() injects mod entries into recipe.pre_exec."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
            "runtime_config": {"mods": ["mods/flash-attn"]},
        })
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])

        # Should have injected one copy dict + one exec string per mod
        assert len(recipe.pre_exec) == 2
        assert isinstance(recipe.pre_exec[0], dict)
        assert "copy" in recipe.pre_exec[0]
        assert isinstance(recipe.pre_exec[1], str)
        assert "run.sh" in recipe.pre_exec[1]

    def test_eugr_prepare_build_failure_raises(self, eugr_builder_with_repo):
        """prepare_image() raises RuntimeError when the build exits non-zero."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
            "runtime_config": {"build_args": ["--flag"]},
        })
        with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=1)
                with pytest.raises(RuntimeError, match="eugr container build failed"):
                    builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])

    def test_eugr_mod_pre_exec_format(self, eugr_builder_with_repo):
        """Injected mod pre_exec entries have correct copy/dest keys and run.sh string."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "eugr-vllm",
            "runtime_config": {"mods": ["mods/flash-attn"]},
        })
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])

        copy_entry = recipe.pre_exec[0]
        exec_entry = recipe.pre_exec[1]

        # The copy dict must have "copy" (source path) and "dest" (container path)
        assert "copy" in copy_entry
        assert "dest" in copy_entry
        assert copy_entry["dest"].startswith("/workspace/mods/")

        # The exec string must reference run.sh
        assert "run.sh" in exec_entry


# ---------------------------------------------------------------------------
# Bootstrap integration — get_builder / list_builders
# ---------------------------------------------------------------------------

class TestBootstrapBuilders:
    """Test builder lookup and listing via sparkrun.core.bootstrap."""

    def test_get_builder_docker_pull(self, v):
        """get_builder('docker-pull') returns a DockerPullBuilder."""
        from sparkrun.builders.docker_pull import DockerPullBuilder
        from sparkrun.core.bootstrap import get_builder
        builder = get_builder("docker-pull", v=v)
        assert isinstance(builder, DockerPullBuilder)

    def test_get_builder_eugr(self, v):
        """get_builder('eugr') returns an EugrBuilder."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.core.bootstrap import get_builder
        builder = get_builder("eugr", v=v)
        assert isinstance(builder, EugrBuilder)

    def test_get_builder_unknown_raises(self, v):
        """get_builder('nonexistent') raises ValueError."""
        from sparkrun.core.bootstrap import get_builder
        with pytest.raises(ValueError, match="nonexistent"):
            get_builder("nonexistent", v=v)

    def test_list_builders(self, v):
        """list_builders() includes both 'docker-pull' and 'eugr'."""
        from sparkrun.core.bootstrap import list_builders
        names = list_builders(v=v)
        assert "docker-pull" in names
        assert "eugr" in names


# ---------------------------------------------------------------------------
# Recipe builder field handling
# ---------------------------------------------------------------------------

class TestRecipeBuilderField:
    """Test that Recipe correctly parses builder / builder_config fields."""

    def test_recipe_builder_field(self):
        """Recipe.from_dict with builder field stores the value."""
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "vllm",
            "builder": "eugr",
        })
        assert recipe.builder == "eugr"

    def test_recipe_builder_config_field(self):
        """Recipe.from_dict with builder_config stores the mapping."""
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "vllm",
            "builder_config": {"key": "value"},
        })
        assert recipe.builder_config == {"key": "value"}

    def test_recipe_builder_not_in_runtime_config(self):
        """builder field is NOT swept into runtime_config."""
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "vllm",
            "builder": "eugr",
        })
        assert "builder" not in recipe.runtime_config

    def test_recipe_builder_config_not_in_runtime_config(self):
        """builder_config field is NOT swept into runtime_config."""
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "vllm",
            "builder_config": {"opt": "val"},
        })
        assert "builder_config" not in recipe.runtime_config

    def test_recipe_builder_in_export(self):
        """builder appears in the export dict when set."""
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "vllm",
            "builder": "eugr",
            "builder_config": {"base": "nvidia/cuda"},
        })
        exported = recipe._build_export_dict()
        assert exported.get("builder") == "eugr"
        assert exported.get("builder_config") == {"base": "nvidia/cuda"}

    def test_recipe_builder_empty_not_in_export(self):
        """Empty builder does NOT appear in the export dict."""
        recipe = Recipe.from_dict({
            "name": "test", "model": "some/model", "runtime": "vllm",
        })
        exported = recipe._build_export_dict()
        assert "builder" not in exported
        assert "builder_config" not in exported
