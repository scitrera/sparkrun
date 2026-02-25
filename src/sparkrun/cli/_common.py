"""Shared CLI infrastructure: utilities, Click types, decorators."""

from __future__ import annotations

import logging
import sys

import click

logger = logging.getLogger(__name__)


# TODO: converge logging with SAF logging
def _setup_logging(verbose: bool):
    """Configure logging based on verbosity.

    Uses explicit handler setup instead of ``logging.basicConfig`` which
    is silently a no-op when the root logger already has handlers (common
    when libraries like ``huggingface_hub`` configure logging on import).
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = ("%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose
           else "%(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers that may have been added by library imports
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)

    from sparkrun.utils import suppress_noisy_loggers
    suppress_noisy_loggers()

    return


def _parse_options(options: tuple[str, ...]) -> dict:
    """Parse --option key=value pairs into a dict.

    Values are auto-coerced to int/float/bool where possible.
    """
    from sparkrun.utils import coerce_value

    result = {}
    for opt in options:
        if "=" not in opt:
            click.echo(
                "Error: --option must be key=value, got: %s" % opt,
                err=True,
            )
            sys.exit(1)
        key, _, value = opt.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            click.echo(
                "Error: --option has empty key: %s" % opt,
                err=True,
            )
            sys.exit(1)
        result[key] = coerce_value(value)
    return result


def _get_config_and_registry(config_path=None):
    """Create SparkrunConfig and RegistryManager."""
    from sparkrun.config import SparkrunConfig
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    registry_mgr = config.get_registry_manager()
    return config, registry_mgr


def _apply_tp_trimming(
        host_list: list[str],
        recipe,
        overrides: dict | None = None,
        tp_override: int | None = None,
) -> list[str]:
    """Trim host list to match tensor_parallel if TP < host count.

    Used by run, stop, and log to ensure they all derive the same
    effective host list (and therefore the same cluster_id).

    Args:
        host_list: Resolved hosts.
        recipe: Loaded recipe (used for defaults).
        overrides: Optional CLI overrides (from --option).
        tp_override: Explicit --tp value (takes precedence).

    Returns:
        Possibly trimmed host list.
    """
    if len(host_list) <= 1:
        return host_list

    if tp_override is not None:
        effective_tp = tp_override
    else:
        config_chain = recipe.build_config_chain(overrides or {})
        tp_val = config_chain.get("tensor_parallel")
        if tp_val is None:
            return host_list
        effective_tp = int(tp_val)

    if effective_tp >= len(host_list):
        return host_list

    trimmed = host_list[:effective_tp]
    logger.info(
        "tensor_parallel=%d < %d hosts; using first %d: %s",
        effective_tp, len(host_list), effective_tp, ", ".join(trimmed),
    )
    return trimmed


def _resolve_cluster_user(
        cluster_name: str | None,
        hosts: str | None,
        hosts_file: str | None,
        cluster_mgr,
) -> str | None:
    """Resolve the SSH user from a cluster definition, if applicable.

    Returns the cluster's configured user, or None if no cluster is
    resolved or the cluster has no user set.
    """
    resolved = cluster_name
    if not resolved and not hosts and not hosts_file:
        resolved = cluster_mgr.get_default() if cluster_mgr else None
    if resolved:
        try:
            cluster_def = cluster_mgr.get(resolved)
            return cluster_def.user
        except Exception:
            logger.debug("Failed to resolve cluster '%s'", resolved, exc_info=True)
    return None


def _get_cluster_manager(v=None):
    """Create a ClusterManager using the SAF config root."""
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root
    # TODO: switch to leveraging scitrera-app-framework plugin for ClusterManager singleton?
    return ClusterManager(get_config_root(v))


def _load_recipe(config, recipe_name):
    """Find, load, and return a recipe.

    Exits with an error message on failure.

    Returns:
        Tuple of (recipe, recipe_path, registry_mgr).
    """
    from sparkrun.recipe import Recipe, find_recipe, RecipeError
    try:
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    return recipe, recipe_path, registry_mgr


def _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v=None):
    """Resolve hosts from CLI args; exit if none are found.

    Returns:
        Tuple of (host_list, cluster_mgr).
    """
    from sparkrun.hosts import resolve_hosts
    cluster_mgr = _get_cluster_manager(v)
    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )
    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts or configure defaults.", err=True)
        sys.exit(1)
    return host_list, cluster_mgr


def _resolve_setup_context(hosts, hosts_file, cluster_name, config, user=None):
    """Resolve hosts, user, and SSH kwargs for setup commands."""
    import os
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    host_list, cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    cluster_user = _resolve_cluster_user(cluster_name, hosts, hosts_file, cluster_mgr)
    if user is None:
        user = cluster_user or config.ssh_user or os.environ.get("USER", "root")
    ssh_kwargs = build_ssh_kwargs(config)
    if user:
        ssh_kwargs["ssh_user"] = user
    return host_list, user, ssh_kwargs


def _display_recipe_detail(recipe, show_vram=True, registry_name=None, cli_overrides=None):
    """Display recipe details (delegates to cli_formatters)."""
    from sparkrun.utils.cli_formatters import display_recipe_detail
    display_recipe_detail(recipe, show_vram=show_vram, registry_name=registry_name, cli_overrides=cli_overrides)


def _display_vram_estimate(recipe, cli_overrides=None, auto_detect=True):
    """Display VRAM estimation (delegates to cli_formatters)."""
    from sparkrun.utils.cli_formatters import display_vram_estimate
    display_vram_estimate(recipe, cli_overrides=cli_overrides, auto_detect=auto_detect)


def _shell_rc_file(shell):
    """Return the RC file path for a given shell name.

    Exits with an error for unsupported shells.
    """
    from pathlib import Path
    home = Path.home()
    rc_files = {
        "bash": home / ".bashrc",
        "zsh": home / ".zshrc",
        "fish": home / ".config" / "fish" / "config.fish",
    }
    if shell not in rc_files:
        click.echo("Error: Unsupported shell: %s" % shell, err=True)
        sys.exit(1)
    return rc_files[shell]


def _detect_shell():
    """Detect the user's login shell, returning (name, rc_file)."""
    import os
    from pathlib import Path

    login_shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in login_shell:
        return "zsh", home / ".zshrc"
    elif "fish" in login_shell:
        return "fish", home / ".config" / "fish" / "config.fish"
    else:
        return "bash", home / ".bashrc"


def _require_uv() -> str:
    """Return path to uv binary, or exit with an error message."""
    import shutil
    # noinspection PyDeprecation
    uv = shutil.which("uv")
    if not uv:
        click.echo("Error: uv is required but not found on PATH.", err=True)
        click.echo("Install uv first: pip install uv", err=True)
        sys.exit(1)
    return uv


class RecipeNameType(click.ParamType):
    """Click parameter type with shell completion for recipe names."""

    name = "recipe"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for recipe names."""
        try:
            from sparkrun.recipe import list_recipes
            config, registry_mgr = _get_config_and_registry()
            # Only read from already-cached registries — no git operations
            recipes = list_recipes(config.get_recipe_search_paths(), registry_mgr)
            return [
                click.shell_completion.CompletionItem(r["file"])
                for r in recipes
                if r["file"].startswith(incomplete)
            ]
        except Exception:
            return []


RECIPE_NAME = RecipeNameType()


class ClusterNameType(click.ParamType):
    """Click parameter type with shell completion for cluster names."""

    name = "cluster"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for cluster names."""
        try:
            mgr = _get_cluster_manager()
            clusters = mgr.list_clusters()
            return [
                click.shell_completion.CompletionItem(c.name)
                for c in clusters
                if c.name.startswith(incomplete)
            ]
        except Exception:
            return []


CLUSTER_NAME = ClusterNameType()


class RegistryNameType(click.ParamType):
    """Click parameter type with shell completion for registry names."""

    name = "registry"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for registry names."""
        try:
            _, registry_mgr = _get_config_and_registry()
            return [
                click.shell_completion.CompletionItem(reg.name)
                for reg in registry_mgr.list_registries()
                if reg.enabled and reg.name.startswith(incomplete)
            ]
        except Exception:
            return []


REGISTRY_NAME = RegistryNameType()


class RuntimeNameType(click.ParamType):
    """Click parameter type with shell completion for runtime names."""

    name = "runtime"

    def shell_complete(self, ctx, param, incomplete):
        """Return completion items for known runtimes."""
        try:
            from sparkrun.recipe import list_recipes
            _, registry_mgr = _get_config_and_registry()
            recipes = list_recipes(registry_manager=registry_mgr)
            runtimes = sorted({r.get("runtime", "") for r in recipes if r.get("runtime")})
            return [
                click.shell_completion.CompletionItem(rt)
                for rt in runtimes
                if rt.startswith(incomplete)
            ]
        except Exception:
            return []


RUNTIME_NAME = RuntimeNameType()


def host_options(f):
    """Common host-targeting options: --hosts, --hosts-file, --cluster."""
    f = click.option("--cluster", "cluster_name", default=None, type=CLUSTER_NAME,
                     help="Use a saved cluster by name")(f)
    f = click.option("--hosts-file", default=None,
                     help="File with hosts (one per line, # comments)")(f)
    f = click.option("--hosts", "-H", default=None,
                     help="Comma-separated host list")(f)
    return f


def dry_run_option(f):
    """Common --dry-run flag."""
    return click.option("--dry-run", "-n", is_flag=True,
                        help="Show what would be done")(f)
