"""Recipe loading, validation, and v1->v2 migration."""

from __future__ import annotations

import logging
import re
from os import path as osp
from pathlib import Path
from typing import Any, TYPE_CHECKING

from vpd.next.util import read_yaml
from vpd.legacy.yaml_dict import vpd_chain, VirtualPathDictChain
from vpd.legacy.arguments import arg_substitute

if TYPE_CHECKING:
    from sparkrun.registry import RegistryManager
    from sparkrun.vram import VRAMEstimate

logger = logging.getLogger(__name__)

_KNOWN_KEYS = {
    "sparkrun_version", "recipe_version", "name", "description", "model",
    "model_revision",
    "runtime", "runtime_version", "mode", "min_nodes", "max_nodes",
    "container", "defaults", "env", "command", "runtime_config",
    "cluster_only", "solo_only",
    "metadata",
}


def _effective_runtime(data: dict[str, Any]) -> str:
    """Determine the effective runtime from raw recipe data.

    Detects eugr-vllm recipes via two signals:
    1. ``recipe_version: "1"`` — the eugr native format.
    2. Presence of ``build_args`` or ``mods`` keys — eugr-specific fields
       that are a dead giveaway even without an explicit version declaration.
    """
    runtime = data.get("runtime", "vllm")
    if runtime != "vllm":
        return runtime
    version = str(data.get("sparkrun_version", data.get("recipe_version", "2")))
    if version == "1":
        return "eugr-vllm"
    if data.get("build_args") or data.get("mods"):
        return "eugr-vllm"
    return runtime


class RecipeError(Exception):
    """Raised when a recipe is invalid or cannot be loaded."""


class Recipe:
    """A loaded and validated sparkrun recipe."""

    def __init__(self, data: dict[str, Any], source_path: str | None = None):
        self._raw = data
        self.source_path = source_path

        # Detect version
        self.sparkrun_version = str(data.get("sparkrun_version", data.get("recipe_version", "2")))

        # Core fields — name defaults to source filename stem if not provided
        default_name = Path(source_path).stem if source_path else "unnamed"
        self.name: str = data.get("name", default_name)
        self.description: str = data.get("description", "")
        self.model: str = data.get("model", "")
        self.model_revision: str | None = data.get("model_revision")
        self.runtime: str = data.get("runtime", "vllm")
        self.runtime_version: str = data.get("runtime_version", "")

        # Topology
        self.mode: str = data.get("mode", "auto")  # "solo", "cluster", "auto"
        self.min_nodes: int = int(data.get("min_nodes", 1))
        self.max_nodes: int | None = data.get("max_nodes")
        if self.mode == 'solo':
            self.max_nodes = self.min_nodes = 1
        elif self.mode == 'auto' and self.min_nodes > 1:
            self.mode = 'cluster'
        elif self.mode == 'auto' and self.max_nodes == 1:
            self.mode = 'solo'

        # Container
        self.container: str = data.get("container", "")

        # Configuration
        self.defaults: dict[str, Any] = dict(data.get("defaults", {}))
        self.env: dict[str, str] = {
            str(k): osp.expandvars(str(v)) for k, v in data.get("env", {}).items()
        }
        self.command: str | None = data.get("command")

        # Metadata section (v2 extension for VRAM estimation, model info)
        raw_metadata = data.get("metadata", {})
        self.metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

        # Metadata values supplement missing top-level fields
        if not self.name or self.name == default_name:
            meta_name = self.metadata.get("name")
            if meta_name:
                self.name = str(meta_name)
        if not self.description:
            meta_desc = self.metadata.get("description")
            if meta_desc:
                self.description = str(meta_desc)

        # Maintainer (metadata-only field)
        self.maintainer: str = str(self.metadata.get("maintainer", ""))

        # Runtime-specific config: explicit runtime_config key takes priority,
        # then unknown top-level keys are auto-swept in.
        self.runtime_config: dict[str, Any] = dict(data.get("runtime_config", {}))
        for k, v in data.items():
            if k not in _KNOWN_KEYS and k not in self.runtime_config:
                self.runtime_config[k] = v

        # v1 compatibility migration
        if self.sparkrun_version == "1":
            self._migrate_v1(data)

        # build_args / mods are eugr-specific fields — a dead giveaway
        # even without an explicit v1 declaration
        if self.runtime == "vllm" and (
            self.runtime_config.get("build_args") or self.runtime_config.get("mods")
        ):
            self.runtime = "eugr-vllm"

        # Handle solo_only/cluster_only as first-class fields (works for both v1 and v2)
        if data.get("cluster_only"):
            self.min_nodes = max(self.min_nodes, 2)
            self.mode = "cluster"
        if data.get("solo_only"):
            self.max_nodes = 1
            self.mode = "solo"

    def _migrate_v1(self, data: dict[str, Any]):
        """Map eugr v1 recipe fields to v2 schema.

        The v1 format is the eugr/spark-vllm-docker native format.
        Recipes declaring ``recipe_version: "1"`` with a vllm target
        should use the ``eugr-vllm`` runtime which delegates to eugr's
        own scripts and container builds.
        """
        if not self.runtime or self.runtime == "vllm":
            self.runtime = "eugr-vllm"

    @property
    def slug(self) -> str:
        """URL/filesystem-safe slug derived from name."""
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get a value from defaults with optional fallback."""
        return self.defaults.get(key, fallback)

    def build_config_chain(self, cli_overrides: dict[str, Any] | None = None,
                           user_config: dict[str, Any] | None = None) -> VirtualPathDictChain:
        """Build cascading config: CLI overrides -> user config -> recipe defaults.

        Also injects 'model' into the chain for template substitution.
        """
        base = dict(self.defaults)
        base.setdefault("model", self.model)
        return vpd_chain(cli_overrides or {}, user_config or {}, base)

    def render_command(self, config_chain: VirtualPathDictChain) -> str | None:
        """Render the command template with values from the config chain.

        Returns None if no command template is defined.
        """
        if not self.command:
            return None

        rendered = self.command.strip()

        # Use vpd arg_substitute for {placeholder} replacement
        # Iterate to handle nested substitutions
        last = None
        while last != rendered:
            last = rendered
            rendered = arg_substitute(rendered, config_chain)

        return rendered

    def validate(self) -> list[str]:
        """Validate the recipe and return a list of warnings/errors."""
        issues = []
        if not self.name:
            issues.append("Recipe missing 'name' field")
        if not self.model:
            issues.append("Recipe missing 'model' field")
        if not self.runtime:
            issues.append("Recipe missing 'runtime' field")
        if self.mode not in ("solo", "cluster", "auto"):
            issues.append("Invalid mode '%s'; expected 'solo', 'cluster', or 'auto'" % self.mode)
        if self.min_nodes < 1:
            issues.append("min_nodes must be >= 1, got %d" % self.min_nodes)
        if self.max_nodes is not None and self.max_nodes < self.min_nodes:
            issues.append("max_nodes (%s) < min_nodes (%s)" % (self.max_nodes, self.min_nodes))

        # Validate metadata if present
        if self.metadata:
            from sparkrun.vram import parse_param_count, bytes_per_element

            mp = self.metadata.get("model_params")
            if mp is not None and parse_param_count(mp) is None:
                issues.append("metadata.model_params %r is not a valid parameter count" % mp)
            md = self.metadata.get("model_dtype")
            if md is not None and bytes_per_element(str(md)) is None:
                issues.append("metadata.model_dtype %r is not a recognized dtype" % md)
            kd = self.metadata.get("kv_dtype")
            if kd is not None and bytes_per_element(str(kd)) is None:
                issues.append("metadata.kv_dtype %r is not a recognized dtype" % kd)

        return issues

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        """Load a recipe from a YAML file path."""
        path = Path(path)
        if not path.exists():
            raise RecipeError("Recipe file not found: %s" % path)
        data = read_yaml(str(path))
        if not isinstance(data, dict):
            raise RecipeError("Recipe file must contain a YAML mapping: %s" % path)
        return cls(data, source_path=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        """Create a recipe from a dict (useful for testing)."""
        return cls(data)

    def estimate_vram(
            self,
            cli_overrides: dict[str, Any] | None = None,
            auto_detect: bool = True,
    ) -> VRAMEstimate:
        """Estimate VRAM usage for this recipe.

        Merges metadata fields with auto-detected HF config (if available).
        CLI overrides for max_model_len, tensor_parallel are respected.

        Args:
            cli_overrides: CLI override values (e.g. tensor_parallel, max_model_len).
            auto_detect: Whether to query HuggingFace Hub for model config.

        Returns:
            VRAMEstimate dataclass with estimation results.
        """
        from sparkrun.vram import (
            VRAMEstimate,
            estimate_vram as _estimate_vram,
            extract_model_info,
            fetch_model_config,
            parse_param_count,
        )

        config = self.build_config_chain(cli_overrides)

        # Start with metadata values
        model_dtype = self.metadata.get("model_dtype")
        model_params_raw = self.metadata.get("model_params")
        kv_dtype = self.metadata.get("kv_dtype")
        num_layers = self.metadata.get("num_layers")
        num_kv_heads = self.metadata.get("num_kv_heads")
        head_dim = self.metadata.get("head_dim")
        model_vram = self.metadata.get("model_vram")
        kv_vram_per_token = self.metadata.get("kv_vram_per_token")

        # Auto-detect from HF if fields are missing and model is specified
        if auto_detect and self.model:
            needs_detection = (
                                      model_vram is None
                                      and (not model_dtype or model_params_raw is None)
                              ) or (
                                      kv_vram_per_token is None
                                      and (not num_layers or not num_kv_heads or not head_dim)
                              )
            if needs_detection:
                hf_config = fetch_model_config(self.model, revision=self.model_revision)
                if hf_config:
                    hf_info = extract_model_info(hf_config)
                    # Fill in missing fields (metadata takes precedence)
                    if not model_dtype:
                        model_dtype = hf_info.get("model_dtype")
                    if not num_layers:
                        num_layers = hf_info.get("num_layers")
                    if not num_kv_heads:
                        num_kv_heads = hf_info.get("num_kv_heads")
                    if not head_dim:
                        head_dim = hf_info.get("head_dim")

        # Parse model_params
        model_params = parse_param_count(model_params_raw) if model_params_raw is not None else None

        # Get effective max_model_len and tensor_parallel from config chain
        max_model_len = config.get("max_model_len")
        if max_model_len is not None:
            max_model_len = int(max_model_len)

        tp_val = config.get("tensor_parallel")
        tensor_parallel = int(tp_val) if tp_val is not None else 1

        # Check for kv_cache_dtype in defaults (runtime-specific)
        if not kv_dtype:
            kv_cache_default = config.get("kv_cache_dtype")
            if kv_cache_default and str(kv_cache_default) != "auto":
                kv_dtype = str(kv_cache_default)

        # GPU memory utilization (runtime budget fraction)
        gpu_mem_val = config.get("gpu_memory_utilization")
        gpu_memory_utilization = float(gpu_mem_val) if gpu_mem_val is not None else None

        return _estimate_vram(
            model_params=model_params,
            model_dtype=str(model_dtype) if model_dtype else None,
            kv_dtype=str(kv_dtype) if kv_dtype else None,
            num_layers=int(num_layers) if num_layers is not None else None,
            num_kv_heads=int(num_kv_heads) if num_kv_heads is not None else None,
            head_dim=int(head_dim) if head_dim is not None else None,
            max_model_len=max_model_len,
            tensor_parallel=tensor_parallel,
            model_vram=float(model_vram) if model_vram is not None else None,
            kv_vram_per_token=float(kv_vram_per_token) if kv_vram_per_token is not None else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def __repr__(self) -> str:
        return "Recipe(name=%r, runtime=%r, model=%r)" % (self.name, self.runtime, self.model)


def find_recipe(name: str, search_paths: list[Path] | None = None,
                registry_manager: RegistryManager | None = None) -> Path:
    """Find a recipe by name across search paths.

    Search order:
    1. Exact/relative file path (if exists)
    2. Given search paths
    3. Registry paths (if registry_manager provided)
    4. Registry file-stem matching (if registry_manager provided)
    """
    # 1. Check if it's a direct path
    direct = Path(name)
    if direct.exists():
        return direct
    # Also try with .yaml extension
    if not name.endswith((".yaml", ".yml")):
        for ext in (".yaml", ".yml"):
            candidate = Path(name + ext)
            if candidate.exists():
                return candidate

    # 2. Search user-provided paths (flat first, then recursive by stem)
    for search_dir in (search_paths or []):
        for ext in ("", ".yaml", ".yml"):
            candidate = search_dir / (name + ext)
            if candidate.exists():
                return candidate
    for search_dir in (search_paths or []):
        for ext in (".yaml", ".yml"):
            for m in search_dir.rglob(f"{name}{ext}"):
                return m

    # 3. Search registry paths (flat first, then recursive by stem)
    if registry_manager:
        for search_dir in registry_manager.get_recipe_paths():
            for ext in ("", ".yaml", ".yml"):
                candidate = search_dir / (name + ext)
                if candidate.exists():
                    return candidate
        for search_dir in registry_manager.get_recipe_paths():
            for ext in (".yaml", ".yml"):
                for m in search_dir.rglob(f"{name}{ext}"):
                    return m

    # 4. Try registry file-stem matching
    if registry_manager:
        matches = registry_manager.find_recipe_in_registries(name)
        if matches:
            # Return first match (user search paths already checked above)
            _registry_name, recipe_path = matches[0]
            return recipe_path

    search_desc = [str(p) for p in (search_paths or [])]
    if registry_manager:
        search_desc.append("registry paths")
    raise RecipeError(
        "Recipe '%s' not found. Searched: %s"
        % (name, search_desc)
    )


def list_recipes(search_paths: list[Path] | None = None,
                 registry_manager: RegistryManager | None = None) -> list[dict[str, str]]:
    """List all available recipes with name and path."""
    recipes = []
    seen_names: set[str] = set()

    all_paths = list(search_paths or [])

    # Add registry paths if available
    if registry_manager:
        all_paths.extend(registry_manager.get_recipe_paths())

    for search_dir in all_paths:
        if not search_dir.is_dir():
            continue

        # Determine if this is a registry path
        registry_name = None
        if registry_manager:
            for reg in registry_manager.list_registries():
                if reg.enabled:
                    reg_path = registry_manager.cache_root / reg.name / reg.subpath
                    if search_dir == reg_path or search_dir.is_relative_to(reg_path):
                        registry_name = reg.name
                        break

        for f in sorted(search_dir.rglob("*.yaml")):
            stem = f.stem
            if stem not in seen_names:
                seen_names.add(stem)
                try:
                    data = read_yaml(str(f))
                    entry = {
                        "name": data.get("name", stem) if isinstance(data, dict) else stem,
                        "file": stem,
                        "path": str(f),
                        "runtime": _effective_runtime(data) if isinstance(data, dict) else "unknown",
                    }
                    if isinstance(data, dict):
                        entry["min_nodes"] = data.get("min_nodes", 1)
                        defaults = data.get("defaults", {})
                        if isinstance(defaults, dict):
                            entry["tp"] = defaults.get("tensor_parallel", "")
                            entry["gpu_mem"] = defaults.get("gpu_memory_utilization", "")
                    if registry_name:
                        entry["registry"] = registry_name
                    recipes.append(entry)
                except Exception:
                    logger.debug("Skipping invalid recipe file: %s", f)
    return recipes
