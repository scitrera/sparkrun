"""Tests for sparkrun.registry module."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml

from sparkrun.registry import (
    DEFAULT_REGISTRIES,
    RegistryEntry,
    RegistryError,
    RegistryManager,
)


@pytest.fixture
def reg_dirs(tmp_path: Path):
    """Create config and cache directories for RegistryManager."""
    config = tmp_path / "config"
    cache = tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    return config, cache


@pytest.fixture
def mgr(reg_dirs):
    """Create a RegistryManager with temp dirs."""
    config, cache = reg_dirs
    return RegistryManager(config, cache)


@pytest.fixture
def sample_entry() -> RegistryEntry:
    """A sample registry entry for testing."""
    return RegistryEntry(
        name="test-registry",
        url="https://github.com/example/repo",
        subpath="recipes",
        description="Test recipes",
    )


@pytest.fixture
def populated_cache(reg_dirs, sample_entry) -> tuple[RegistryManager, Path]:
    """Create a RegistryManager with a fake cached registry containing recipe files."""
    config, cache = reg_dirs
    mgr = RegistryManager(config, cache)

    # Save the sample registry to config
    mgr._save_registries([sample_entry])

    # Create fake cached repo with recipes
    recipe_dir = cache / sample_entry.name / sample_entry.subpath
    recipe_dir.mkdir(parents=True)
    # Also create a .git dir to simulate a cloned repo
    (cache / sample_entry.name / ".git").mkdir()

    # Create test recipe files
    recipe1 = {
        "sparkrun_version": "2",
        "name": "Test vLLM Recipe",
        "description": "A test recipe for vLLM inference",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "container": "scitrera/dgx-spark-vllm:latest",
    }
    recipe2 = {
        "sparkrun_version": "2",
        "name": "Test SGLang Recipe",
        "description": "A test recipe for SGLang inference",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "container": "scitrera/dgx-spark-sglang:latest",
    }
    with open(recipe_dir / "test-vllm.yaml", "w") as f:
        yaml.dump(recipe1, f)
    with open(recipe_dir / "test-sglang.yaml", "w") as f:
        yaml.dump(recipe2, f)

    return mgr, recipe_dir


class TestRegistryEntry:
    """Test RegistryEntry dataclass."""

    def test_default_values(self):
        """Test that RegistryEntry has sensible defaults."""
        entry = RegistryEntry(name="test", url="https://example.com", subpath="recipes")
        assert entry.description == ""
        assert entry.enabled is True

    def test_all_fields(self):
        """Test creating entry with all fields."""
        entry = RegistryEntry(
            name="test",
            url="https://example.com",
            subpath="recipes/sub",
            description="Test registry",
            enabled=False,
        )
        assert entry.name == "test"
        assert entry.url == "https://example.com"
        assert entry.subpath == "recipes/sub"
        assert entry.description == "Test registry"
        assert entry.enabled is False


class TestDefaultRegistries:
    """Test DEFAULT_REGISTRIES."""

    def test_has_official_registry(self):
        """Test that DEFAULT_REGISTRIES includes the sparkrun-official registry."""
        assert len(DEFAULT_REGISTRIES) >= 1
        official = DEFAULT_REGISTRIES[0]
        assert official.name == "sparkrun-official"
        assert "github.com/scitrera/oss-spark-run" in official.url
        assert official.enabled is True

    def test_has_eugr_vllm_registry(self):
        """Test that DEFAULT_REGISTRIES includes the eugr-vllm registry."""
        assert len(DEFAULT_REGISTRIES) >= 2
        eugr = DEFAULT_REGISTRIES[1]
        assert eugr.name == "eugr-vllm"
        assert "github.com/eugr/spark-vllm-docker" in eugr.url
        assert eugr.enabled is True

    def test_only_two_default_registries(self):
        """Test that there are exactly two default registries."""
        assert len(DEFAULT_REGISTRIES) == 2

    def test_official_registry_subpath(self):
        """Test the sparkrun-official registry subpath."""
        official = DEFAULT_REGISTRIES[0]
        assert official.subpath == "recipes"


class TestRegistryManagerInit:
    """Test RegistryManager initialization."""

    def test_creates_directories(self, tmp_path: Path):
        """Test that init creates config and cache directories."""
        config = tmp_path / "new_config"
        cache = tmp_path / "new_cache"
        mgr = RegistryManager(config, cache)
        assert config.exists()
        assert cache.exists()

    def test_default_cache_root(self, tmp_path: Path):
        """Test that cache_root defaults when not provided."""
        mgr = RegistryManager(tmp_path)
        assert mgr.cache_root == Path.home() / ".cache/sparkrun/registries"

    def test_registries_path(self, mgr):
        """Test the registries.yaml path property."""
        assert mgr._registries_path.name == "registries.yaml"


class TestRegistryCRUD:
    """Test registry add/remove/list/get operations."""

    def test_list_defaults_when_no_config(self, mgr):
        """Test that list_registries returns defaults when no config file exists."""
        registries = mgr.list_registries()
        assert len(registries) == len(DEFAULT_REGISTRIES)
        assert registries[0].name == "sparkrun-official"

    def test_add_registry(self, mgr, sample_entry):
        """Test adding a new registry."""
        mgr.add_registry(sample_entry)
        registries = mgr.list_registries()
        names = [r.name for r in registries]
        assert sample_entry.name in names

    def test_add_duplicate_raises(self, mgr, sample_entry):
        """Test that adding a duplicate registry raises RegistryError."""
        mgr.add_registry(sample_entry)
        with pytest.raises(RegistryError, match="already exists"):
            mgr.add_registry(sample_entry)

    def test_remove_registry(self, mgr, sample_entry):
        """Test removing a registry."""
        mgr.add_registry(sample_entry)
        mgr.remove_registry(sample_entry.name)
        registries = mgr.list_registries()
        names = [r.name for r in registries]
        assert sample_entry.name not in names

    def test_remove_nonexistent_raises(self, mgr):
        """Test that removing a nonexistent registry raises RegistryError."""
        with pytest.raises(RegistryError, match="not found"):
            mgr.remove_registry("nonexistent")

    def test_get_registry(self, mgr, sample_entry):
        """Test getting a registry by name."""
        mgr.add_registry(sample_entry)
        retrieved = mgr.get_registry(sample_entry.name)
        assert retrieved.name == sample_entry.name
        assert retrieved.url == sample_entry.url

    def test_get_nonexistent_raises(self, mgr):
        """Test that getting a nonexistent registry raises RegistryError."""
        with pytest.raises(RegistryError, match="not found"):
            mgr.get_registry("nonexistent")


class TestRegistrySaveLoad:
    """Test registry persistence via YAML."""

    def test_save_and_load_roundtrip(self, mgr, sample_entry):
        """Test that registries survive save/load cycle."""
        entries = [sample_entry]
        mgr._save_registries(entries)
        loaded = mgr._load_registries()
        assert len(loaded) == 1
        assert loaded[0].name == sample_entry.name
        assert loaded[0].url == sample_entry.url
        assert loaded[0].subpath == sample_entry.subpath

    def test_save_creates_yaml_file(self, mgr, sample_entry):
        """Test that _save_registries creates the YAML file."""
        mgr._save_registries([sample_entry])
        assert mgr._registries_path.exists()

    def test_load_with_disabled_registry(self, mgr):
        """Test loading a registry with enabled=false."""
        entry = RegistryEntry(
            name="disabled",
            url="https://example.com",
            subpath="recipes",
            enabled=False,
        )
        mgr._save_registries([entry])
        loaded = mgr._load_registries()
        assert loaded[0].enabled is False


class TestRegistryCache:
    """Test cache directory management."""

    def test_cache_dir_path(self, mgr):
        """Test that _cache_dir returns correct path."""
        path = mgr._cache_dir("test-registry")
        assert path == mgr.cache_root / "test-registry"

    def test_recipe_dir_returns_none_when_not_cached(self, mgr, sample_entry):
        """Test that _recipe_dir returns None when cache doesn't exist."""
        assert mgr._recipe_dir(sample_entry) is None

    def test_recipe_dir_returns_path_when_cached(self, mgr, sample_entry):
        """Test that _recipe_dir returns path when cache exists."""
        # Create fake cache
        recipe_dir = mgr._cache_dir(sample_entry.name) / sample_entry.subpath
        recipe_dir.mkdir(parents=True)
        result = mgr._recipe_dir(sample_entry)
        assert result == recipe_dir


class TestRegistryUpdate:
    """Test registry update (git clone/pull) operations."""

    def test_update_calls_clone_for_new(self, mgr, sample_entry):
        """Test that update clones a new registry."""
        mgr._save_registries([sample_entry])
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            mgr.update(sample_entry.name)
            # Should call git clone
            calls = mock_run.call_args_list
            assert any("clone" in str(c) for c in calls)

    def test_update_calls_pull_for_existing(self, mgr, sample_entry):
        """Test that update pulls an existing registry."""
        mgr._save_registries([sample_entry])
        # Create fake .git dir to simulate existing clone
        cache_dir = mgr._cache_dir(sample_entry.name)
        (cache_dir / ".git").mkdir(parents=True)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            mgr.update(sample_entry.name)
            # Should call git pull
            calls = mock_run.call_args_list
            assert any("pull" in str(c) for c in calls)

    def test_update_all_registries(self, mgr, sample_entry):
        """Test that update() with no name updates all enabled registries."""
        second = RegistryEntry(
            name="second",
            url="https://example.com/2",
            subpath="recipes",
        )
        mgr._save_registries([sample_entry, second])
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            mgr.update()
            # Should have clone calls for both
            assert mock_run.call_count >= 2

    def test_update_skips_disabled(self, mgr):
        """Test that update skips disabled registries."""
        disabled = RegistryEntry(
            name="disabled",
            url="https://example.com",
            subpath="recipes",
            enabled=False,
        )
        mgr._save_registries([disabled])
        with mock.patch("subprocess.run") as mock_run:
            mgr.update()
            mock_run.assert_not_called()

    def test_clone_failure_is_logged_not_raised(self, mgr, sample_entry):
        """Test that git clone failure is logged but doesn't raise."""
        mgr._save_registries([sample_entry])
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="fatal: error")
            # Should not raise
            mgr.update(sample_entry.name)

    def test_clone_timeout_is_logged_not_raised(self, mgr, sample_entry):
        """Test that git timeout is logged but doesn't raise."""
        import subprocess as sp

        mgr._save_registries([sample_entry])
        with mock.patch("subprocess.run", side_effect=sp.TimeoutExpired("git", 60)):
            # Should not raise
            mgr._clone_or_pull(sample_entry)


class TestEnsureInitialized:
    """Test ensure_initialized auto-download behavior."""

    def test_calls_update_when_no_cache(self, mgr, sample_entry):
        """Test that ensure_initialized calls update when no cache exists."""
        mgr._save_registries([sample_entry])
        with mock.patch.object(mgr, "update") as mock_update:
            mgr.ensure_initialized()
            mock_update.assert_called_once()

    def test_skips_when_cache_exists(self, mgr, sample_entry):
        """Test that ensure_initialized skips when cache already exists."""
        mgr._save_registries([sample_entry])
        # Create fake .git dir
        (mgr._cache_dir(sample_entry.name) / ".git").mkdir(parents=True)
        with mock.patch.object(mgr, "update") as mock_update:
            mgr.ensure_initialized()
            mock_update.assert_not_called()


class TestRecipeDiscovery:
    """Test recipe path and search functionality."""

    def test_get_recipe_paths_empty(self, mgr):
        """Test get_recipe_paths with no cached registries."""
        paths = mgr.get_recipe_paths()
        assert paths == []

    def test_get_recipe_paths_with_cache(self, populated_cache):
        """Test get_recipe_paths returns cached recipe directories."""
        mgr, recipe_dir = populated_cache
        paths = mgr.get_recipe_paths()
        assert len(paths) == 1
        assert paths[0] == recipe_dir

    def test_get_recipe_paths_skips_disabled(self, reg_dirs):
        """Test that get_recipe_paths skips disabled registries."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)
        entry = RegistryEntry(
            name="disabled",
            url="https://example.com",
            subpath="recipes",
            enabled=False,
        )
        mgr._save_registries([entry])
        # Create the cache anyway
        recipe_dir = cache / "disabled" / "recipes"
        recipe_dir.mkdir(parents=True)
        paths = mgr.get_recipe_paths()
        assert paths == []

    def test_search_recipes_by_name(self, populated_cache):
        """Test searching recipes by name."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("vLLM")
        assert len(results) >= 1
        assert any("vLLM" in r["name"] for r in results)

    def test_search_recipes_by_model(self, populated_cache):
        """Test searching recipes by model name."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("llama")
        assert len(results) >= 2  # Both recipes use llama

    def test_search_recipes_by_file_stem(self, populated_cache):
        """Test searching recipes by file stem."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("test-vllm")
        assert len(results) >= 1

    def test_search_recipes_case_insensitive(self, populated_cache):
        """Test that recipe search is case-insensitive."""
        mgr, _ = populated_cache
        upper = mgr.search_recipes("VLLM")
        lower = mgr.search_recipes("vllm")
        assert len(upper) == len(lower)

    def test_search_recipes_no_results(self, populated_cache):
        """Test that search returns empty list for no matches."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("nonexistent-model-xyz")
        assert results == []

    def test_search_results_have_registry_field(self, populated_cache):
        """Test that search results include registry name."""
        mgr, _ = populated_cache
        results = mgr.search_recipes("llama")
        for r in results:
            assert "registry" in r
            assert r["registry"] == "test-registry"

    def test_find_recipe_in_registries(self, populated_cache):
        """Test finding a recipe by file stem."""
        mgr, _ = populated_cache
        matches = mgr.find_recipe_in_registries("test-vllm")
        assert len(matches) == 1
        registry_name, path = matches[0]
        assert registry_name == "test-registry"
        assert path.name == "test-vllm.yaml"

    def test_find_recipe_not_found(self, populated_cache):
        """Test that find returns empty for nonexistent recipe."""
        mgr, _ = populated_cache
        matches = mgr.find_recipe_in_registries("nonexistent-recipe")
        assert matches == []

    def test_find_recipe_multiple_registries(self, reg_dirs):
        """Test finding a recipe that exists in multiple registries."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)

        entries = [
            RegistryEntry(name="reg1", url="https://example.com/1", subpath="recipes"),
            RegistryEntry(name="reg2", url="https://example.com/2", subpath="recipes"),
        ]
        mgr._save_registries(entries)

        # Create same recipe in both registries
        for entry in entries:
            recipe_dir = cache / entry.name / entry.subpath
            recipe_dir.mkdir(parents=True)
            (cache / entry.name / ".git").mkdir(exist_ok=True)
            with open(recipe_dir / "shared-recipe.yaml", "w") as f:
                yaml.dump({"name": "Shared Recipe", "model": "test"}, f)

        matches = mgr.find_recipe_in_registries("shared-recipe")
        assert len(matches) == 2
        registry_names = {m[0] for m in matches}
        assert registry_names == {"reg1", "reg2"}

    def test_find_recipe_in_subdirectory_fallback(self, reg_dirs):
        """Test finding a recipe in a nested subdirectory when no flat match exists."""
        config, cache = reg_dirs
        mgr = RegistryManager(config, cache)

        entry = RegistryEntry(
            name="nested-registry",
            url="https://example.com/nested",
            subpath="recipes",
        )
        mgr._save_registries([entry])

        # Create registry with recipes ONLY in a subdirectory (no flat recipes)
        recipe_dir = cache / entry.name / entry.subpath
        nested_dir = recipe_dir / "qwen3"
        nested_dir.mkdir(parents=True)
        (cache / entry.name / ".git").mkdir(exist_ok=True)

        # Create recipe in subdirectory
        nested_recipe = {
            "sparkrun_version": "2",
            "name": "Qwen3 vLLM Recipe",
            "description": "A nested test recipe",
            "model": "Qwen/Qwen3-1.7b",
            "runtime": "vllm",
            "container": "scitrera/dgx-spark-vllm:latest",
        }
        with open(nested_dir / "qwen3-1.7b-vllm.yaml", "w") as f:
            yaml.dump(nested_recipe, f)

        # Should find the recipe by stem even though it's in a subdirectory
        matches = mgr.find_recipe_in_registries("qwen3-1.7b-vllm")
        assert len(matches) == 1
        registry_name, path = matches[0]
        assert registry_name == "nested-registry"
        assert path.name == "qwen3-1.7b-vllm.yaml"
        assert "qwen3" in str(path)  # Verify it's in the subdirectory
