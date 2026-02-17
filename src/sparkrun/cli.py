"""sparkrun CLI — launch inference workloads on DGX Spark."""

from __future__ import annotations

import logging
import sys

import click

from sparkrun import __version__

logger = logging.getLogger(__name__)


def _coerce_value(value: str):
    """Auto-coerce a string value to int, float, or bool if appropriate."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_options(options: tuple[str, ...]) -> dict:
    """Parse --option key=value pairs into a dict.

    Values are auto-coerced to int/float/bool where possible.
    """
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
        result[key] = _coerce_value(value)
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
            from sparkrun.cluster_manager import ClusterManager
            from sparkrun.config import get_config_root
            mgr = ClusterManager(get_config_root())
            clusters = mgr.list_clusters()
            return [
                click.shell_completion.CompletionItem(c.name)
                for c in clusters
                if c.name.startswith(incomplete)
            ]
        except Exception:
            return []


CLUSTER_NAME = ClusterNameType()


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

    # Suppress noisy HTTP loggers (huggingface_hub uses httpx)
    for name in ("httpx", "httpcore.http11", "httpcore.connection",
                 "urllib3.connectionpool", "huggingface_hub.file_download"):
        logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose/debug output")
@click.version_option(__version__, prog_name="sparkrun")
@click.pass_context
def main(ctx, verbose):
    """sparkrun — Launch inference workloads on NVIDIA DGX Spark systems."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list (first=head)")
@click.option("--hosts-file", default=None, help="File with hosts (one per line, # comments)")
@click.option("--cluster", "cluster_name", default=None, help="Use a saved cluster by name")
@click.option("--solo", is_flag=True, help="Force single-node mode")
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None, help="Override GPU memory utilization")
@click.option("--image", default=None, help="Override container image")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory")
@click.option("--ray-port", type=int, default=46379, help="Ray GCS port (vLLM)")
@click.option("--init-port", type=int, default=25000, help="SGLang distributed init port")
@click.option("--dashboard", is_flag=True, help="Enable Ray dashboard on head node")
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port")
# @click.option("--setup", is_flag=True, hidden=True, help="Deprecated: distribution is now automatic")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--no-follow", is_flag=True, help="Don't follow container logs after launch")
@click.option("--skip-ib", is_flag=True, help="Skip InfiniBand detection")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.option("--option", "-o", "options", multiple=True,
              help="Override any recipe default: -o key=value (repeatable)")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def run(
        ctx, recipe_name, hosts, hosts_file, cluster_name, solo, port, tensor_parallel,
        gpu_mem, image, cache_dir, ray_port, init_port, dashboard, dashboard_port,
        dry_run, foreground, no_follow, skip_ib, options, extra_args, config_path=None, setup=True,
):
    """Run an inference recipe.

    RECIPE_NAME can be a recipe file path or a name to search for.

    Examples:

      sparkrun run glm-4.7-flash-awq --solo

      sparkrun run glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      sparkrun run glm-4.7-flash-awq --cluster mylab

      sparkrun run my-recipe.yaml --port 9000 --gpu-mem 0.8

      sparkrun run my-recipe.yaml -o attention_backend=triton -o max_model_len=4096
    """
    from sparkrun.bootstrap import init_sparkrun, get_runtime
    from sparkrun.recipe import Recipe, find_recipe, RecipeError
    from sparkrun.config import SparkrunConfig

    v = init_sparkrun()
    # SAF's init_framework_desktop reconfigures the root logger — re-apply ours
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Find and load recipe
    try:
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Validate recipe
    issues = recipe.validate()
    if issues:
        for issue in issues:
            click.echo(f"Warning: {issue}", err=True)

    # Build overrides from --option flags first (lowest priority)
    overrides = _parse_options(options)
    # Dedicated CLI params override --option values
    if port is not None:
        overrides["port"] = port
    if tensor_parallel is not None:
        overrides["tensor_parallel"] = tensor_parallel
    if gpu_mem is not None:
        overrides["gpu_memory_utilization"] = gpu_mem
    if image:
        recipe.container = image

    # Resolve runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Runtime-specific validation
    runtime_issues = runtime.validate_recipe(recipe)
    for issue in runtime_issues:
        click.echo(f"Warning: {issue}", err=True)

    # Determine hosts
    from sparkrun.hosts import resolve_hosts
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root

    cluster_mgr = ClusterManager(get_config_root(v))
    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )

    # Determine host source for display
    if hosts:
        host_source = "--hosts"
    elif hosts_file:
        host_source = f"hosts file ({hosts_file})"
    elif cluster_name:
        host_source = f"cluster '{cluster_name}'"
    else:
        default_name = cluster_mgr.get_default() if cluster_mgr else None
        if default_name:
            host_source = f"default cluster '{default_name}'"
        elif config.default_hosts:
            host_source = "config defaults"
        else:
            host_source = "localhost"

    # Validate tensor_parallel vs host count
    # On DGX Spark each host has 1 GPU, so tensor_parallel maps to node count.
    if len(host_list) > 1 and not solo:
        config_chain = recipe.build_config_chain(overrides)
        tp_val = config_chain.get("tensor_parallel")
        if tp_val is not None:
            effective_tp = int(tp_val)
            if effective_tp > len(host_list):
                click.echo(
                    "Error: tensor_parallel=%d requires %d hosts, but only %d provided"
                    % (effective_tp, effective_tp, len(host_list)),
                    err=True,
                )
                sys.exit(1)
            elif effective_tp < len(host_list):
                original_count = len(host_list)
                host_list = _apply_tp_trimming(host_list, recipe, overrides)
                click.echo(
                    "Note: tensor_parallel=%d, using %d of %d hosts"
                    % (effective_tp, effective_tp, original_count)
                )

    # Enforce max_nodes: trim host list if recipe caps node count.
    # Must happen before cluster_id derivation so stop/logs match.
    if recipe.max_nodes is not None and len(host_list) > recipe.max_nodes:
        click.echo(
            "Note: recipe max_nodes=%d, using %d of %d hosts"
            % (recipe.max_nodes, recipe.max_nodes, len(host_list))
        )
        host_list = host_list[:recipe.max_nodes]

    # Determine mode
    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "cluster" and is_solo and not solo:
        click.echo("Warning: Recipe requires cluster mode but only one host specified", err=True)
    if recipe.mode == "solo":
        is_solo = True
        host_list = host_list[:1]

    # Derive deterministic cluster_id from recipe + (trimmed) hosts
    from sparkrun.orchestration.docker import generate_cluster_id
    cluster_id = generate_cluster_id(recipe, host_list)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    # Distribution phase: ensure image/model locally, distribute to hosts,
    # detect IB for NCCL env + fast transfer routing.
    # Always runs for non-delegating runtimes (hash checks make it cheap
    # when resources are already present on all hosts).
    nccl_env = None
    effective_cache_dir = cache_dir or str(config.hf_cache_dir)
    if not runtime.is_delegating_runtime():
        nccl_env = _distribute_resources(
            container_image, recipe.model, host_list,
            effective_cache_dir,
            config, dry_run, skip_ib,
            model_revision=recipe.model_revision,
        )

    # For GGUF models that were pre-synced, resolve the container-internal
    # cache path and inject it as the ``model`` override so the recipe
    # command template renders ``{model}`` with the local path instead
    # of the HF repo spec (which would re-download at serve time).
    from sparkrun.models.download import is_gguf_model, resolve_gguf_container_path
    if is_gguf_model(recipe.model) and not dry_run:
        gguf_container_path = resolve_gguf_container_path(
            recipe.model, effective_cache_dir,
        )
        if gguf_container_path:
            overrides["_gguf_model_path"] = gguf_container_path
            overrides["model"] = gguf_container_path
            logger.info("GGUF model pre-synced, container path: %s", gguf_container_path)

    # Generate serve command for display
    serve_command = runtime.generate_command(
        recipe=recipe,
        overrides=overrides,
        is_cluster=not is_solo,
        num_nodes=len(host_list),
        head_ip=None,  # determined during launch
    )

    # Display summary
    click.echo(f"Runtime:   {runtime.runtime_name}")
    click.echo(f"Image:     {container_image}")
    click.echo(f"Model:     {recipe.model}")
    click.echo(f"Cluster:   {cluster_id}")
    if is_solo:
        click.echo("Mode:      solo")
    else:
        click.echo(f"Mode:      cluster ({len(host_list)} nodes)")

    _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True)

    click.echo()
    click.echo(f"Hosts:     {host_source}")
    if is_solo:
        target = host_list[0] if host_list else "localhost"
        click.echo(f"  Target:  {target}")
    else:
        click.echo(f"  Head:    {host_list[0]}")
        if len(host_list) > 1:
            click.echo(f"  Workers: {', '.join(host_list[1:])}")

    click.echo()
    click.echo("Serve command:")
    for line in serve_command.strip().splitlines():
        click.echo(f"  {line}")
    click.echo()

    # Launch — the runtime controls solo vs cluster orchestration.
    # If distribution pre-detected IB, pass nccl_env through to avoid
    # redundant detection inside the runtime.
    rc = runtime.run(
        hosts=host_list,
        image=container_image,
        serve_command=serve_command,
        recipe=recipe,
        overrides=overrides,
        cluster_id=cluster_id,
        env=recipe.env,
        cache_dir=cache_dir or str(config.hf_cache_dir),
        config=config,
        dry_run=dry_run,
        detached=not foreground,
        skip_ib_detect=nccl_env is not None or skip_ib,
        nccl_env=nccl_env,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        dashboard=dashboard,
        init_port=init_port,
    )

    # Follow container logs after a successful detached launch
    if rc == 0 and not foreground and not dry_run and not no_follow:
        runtime.follow_logs(
            hosts=host_list,
            cluster_id=cluster_id,
            config=config,
            dry_run=dry_run,
        )

    sys.exit(rc)


@main.command("list")
@click.option("--registry", default=None, help="Filter by registry name")
@click.argument("query", required=False)
@click.pass_context
def list_cmd(ctx, registry, query):
    """List available recipes (alias for 'recipe list')."""
    ctx.invoke(recipe_list, registry=registry, query=query)


def _display_recipe_detail(recipe, show_vram=True, registry_name=None):
    """Display recipe details (shared by show and recipe show commands)."""
    click.echo(f"Name:         {recipe.name}")
    click.echo(f"Description:  {recipe.description}")
    if recipe.maintainer:
        click.echo(f"Maintainer:   {recipe.maintainer}")
    click.echo(f"Runtime:      {recipe.runtime}")
    click.echo(f"Model:        {recipe.model}")
    click.echo(f"Container:    {recipe.container}")
    max_nodes = recipe.max_nodes or "unlimited"
    click.echo(f"Nodes:        {recipe.min_nodes} - {max_nodes}")
    click.echo(f"Repository:   {registry_name or 'Local'}")
    click.echo(f"File Path:    {recipe.source_path}")

    if recipe.defaults:
        click.echo("\nDefaults:")
        for k, v in sorted(recipe.defaults.items()):
            click.echo(f"  {k}: {v}")

    if recipe.env:
        click.echo("\nEnvironment:")
        for k, v in sorted(recipe.env.items()):
            click.echo(f"  {k}={v}")

    if recipe.command:
        click.echo(f"\nCommand:\n  {recipe.command.strip()}")

    if show_vram:
        _display_vram_estimate(recipe)


def _display_vram_estimate(recipe, cli_overrides=None, auto_detect=True):
    """Display VRAM estimation for a recipe."""
    from sparkrun.vram import DGX_SPARK_VRAM_GB

    try:
        est = recipe.estimate_vram(cli_overrides=cli_overrides, auto_detect=auto_detect)
    except Exception as e:
        click.echo(f"\nVRAM estimation failed: {e}", err=True)
        return

    click.echo("\nVRAM Estimation:")
    if est.model_dtype:
        click.echo(f"  Model dtype:      {est.model_dtype}")
    if est.model_params:
        click.echo(f"  Model params:     {est.model_params:,}")
    click.echo(f"  KV cache dtype:   {est.kv_dtype or 'bfloat16 (default)'}")
    if all([est.num_layers, est.num_kv_heads, est.head_dim]):
        click.echo(f"  Architecture:     {est.num_layers} layers, {est.num_kv_heads} KV heads, {est.head_dim} head_dim")
    click.echo(f"  Model weights:    {est.model_weights_gb:.2f} GB")
    if est.kv_cache_total_gb is not None:
        click.echo(f"  KV cache:         {est.kv_cache_total_gb:.2f} GB (max_model_len={est.max_model_len:,})")
    click.echo(f"  Tensor parallel:  {est.tensor_parallel}")
    click.echo(f"  Per-GPU total:    {est.total_per_gpu_gb:.2f} GB")
    fit_str = "YES" if est.fits_dgx_spark else "EXCEEDS %.0f GB" % DGX_SPARK_VRAM_GB
    click.echo(f"  DGX Spark fit:    {fit_str}")

    # GPU memory budget analysis
    if est.gpu_memory_utilization is not None:
        click.echo(f"\n  GPU Memory Budget:")
        click.echo(f"    gpu_memory_utilization: {est.gpu_memory_utilization:.0%}")
        click.echo(f"    Usable GPU memory:     {est.usable_gpu_memory_gb:.1f} GB"
                   f" ({DGX_SPARK_VRAM_GB:.0f} GB x {est.gpu_memory_utilization:.0%})")
        click.echo(f"    Available for KV:      {est.available_kv_gb:.1f} GB")
        if est.max_context_tokens is not None:
            click.echo(f"    Max context tokens:    {est.max_context_tokens:,}")
            if est.context_multiplier is not None and est.max_model_len:
                click.echo(f"    Context multiplier:    {est.context_multiplier:.1f}x"
                           f" (vs max_model_len={est.max_model_len:,})")
                if est.context_multiplier < 1.0:
                    click.echo(f"    WARNING: max_model_len exceeds available KV budget"
                               f" ({est.context_multiplier:.1%} fits)")

    for w in est.warnings:
        click.echo(f"  Warning: {w}")


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.pass_context
def show(ctx, recipe_name, no_vram):
    """Show detailed recipe information (alias for 'recipe show')."""
    ctx.invoke(recipe_show, recipe_name=recipe_name, no_vram=no_vram)


@main.command("search")
@click.argument("query")
@click.pass_context
def search_cmd(ctx, query):
    """Search for recipes by name, model, or description (alias for 'recipe search')."""
    ctx.invoke(recipe_search, query=query)


# ---------------------------------------------------------------------------
# setup group
# ---------------------------------------------------------------------------

@main.group()
@click.pass_context
def setup(ctx):
    """Setup and configuration commands."""
    pass


@setup.command("completion")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None,
              help="Shell type (auto-detected if not specified)")
@click.pass_context
def setup_completion(ctx, shell):
    """Install shell tab-completion for sparkrun.

    Detects your current shell and appends the completion setup to
    your shell config file (~/.bashrc, ~/.zshrc, or ~/.config/fish/...).

    Examples:

      sparkrun setup completion

      sparkrun setup completion --shell bash
    """
    from pathlib import Path

    if not shell:
        shell, rc_file = _detect_shell()
    else:
        home = Path.home()
        if shell == "bash":
            rc_file = home / ".bashrc"
        elif shell == "zsh":
            rc_file = home / ".zshrc"
        elif shell == "fish":
            rc_file = home / ".config" / "fish" / "config.fish"
        else:
            click.echo("Error: Unsupported shell: %s" % shell, err=True)
            sys.exit(1)

    completion_var = "_SPARKRUN_COMPLETE"

    if shell == "bash":
        snippet = 'eval "$(%s=bash_source sparkrun)"' % completion_var
    elif shell == "zsh":
        snippet = 'eval "$(%s=zsh_source sparkrun)"' % completion_var
    elif shell == "fish":
        snippet = "%s=fish_source sparkrun | source" % completion_var

    # Check if already installed
    if rc_file.exists():
        contents = rc_file.read_text()
        if completion_var in contents:
            click.echo("Completion already configured in %s" % rc_file)
            return

    # Ensure parent directory exists (for fish)
    rc_file.parent.mkdir(parents=True, exist_ok=True)

    with open(rc_file, "a") as f:
        f.write("\n# sparkrun tab-completion\n")
        f.write(snippet + "\n")

    click.echo("Completion installed for %s in %s" % (shell, rc_file))
    click.echo("Restart your shell or run: source %s" % rc_file)


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


@setup.command("install")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None,
              help="Shell type (auto-detected if not specified)")
@click.pass_context
def setup_install(ctx, shell):
    """Install sparkrun and tab-completion.

    Requires uv (https://docs.astral.sh/uv/).  Typical usage:

    \b
      uvx sparkrun setup install

    This installs sparkrun as a uv tool (real binary on PATH), cleans up
    any old aliases/functions from previous installs, and configures
    tab-completion.
    """
    import subprocess
    from pathlib import Path

    if not shell:
        shell, rc_file = _detect_shell()
    else:
        home = Path.home()
        if shell == "bash":
            rc_file = home / ".bashrc"
        elif shell == "zsh":
            rc_file = home / ".zshrc"
        elif shell == "fish":
            rc_file = home / ".config" / "fish" / "config.fish"
        else:
            click.echo("Error: Unsupported shell: %s" % shell, err=True)
            sys.exit(1)

    # Step 1: Install sparkrun via uv tool
    uv = _require_uv()

    click.echo("Installing sparkrun via uv tool install...")
    result = subprocess.run(
        [uv, "tool", "install", "sparkrun", "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo("Error installing sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)
    click.echo("sparkrun installed on PATH")

    # Step 2: Clean up old aliases/functions from previous installs
    if rc_file.exists():
        old_markers = [
            "alias sparkrun=", "alias sparkrun ",
            "function sparkrun", "sparkrun()",
        ]
        contents = rc_file.read_text()
        lines = contents.splitlines(keepends=True)
        cleaned = [ln for ln in lines if not any(m in ln for m in old_markers)]
        if len(cleaned) != len(lines):
            rc_file.write_text("".join(cleaned))
            click.echo("Removed old sparkrun alias/function from %s" % rc_file)

    # Step 3: Set up tab-completion
    ctx.invoke(setup_completion, shell=shell)

    click.echo()
    click.echo("Restart your shell or run: source %s" % rc_file)


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


@setup.command("update")
@click.pass_context
def setup_update(ctx):
    """Update sparkrun to the latest version.

    Runs ``uv tool upgrade sparkrun`` to fetch the latest release.
    Only works when sparkrun was installed via ``uv tool install``.
    Shows whether an update was available or the current version is
    already the latest.
    """
    import subprocess

    from sparkrun import __version__ as old_version

    uv = _require_uv()

    # Guard: only upgrade if sparkrun was installed via uv tool
    check = subprocess.run(
        [uv, "tool", "list"],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or "sparkrun" not in check.stdout:
        click.echo(
            "Error: sparkrun was not installed via 'uv tool install'.\n"
            "Cannot safely upgrade — manage updates through your package manager instead.",
            err=True,
        )
        sys.exit(1)

    click.echo("Checking for updates (current: %s)..." % old_version)
    result = subprocess.run(
        [uv, "tool", "upgrade", "sparkrun"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo("Error updating sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)

    # The running process still has the old module cached, and reload
    # won't help because uv tool installs into a separate virtualenv.
    # Ask the newly installed binary instead.
    ver_result = subprocess.run(
        ["sparkrun", "--version"],
        capture_output=True, text=True,
    )
    if ver_result.returncode == 0:
        new_version = ver_result.stdout.strip().rsplit(None, 1)[-1]
        if new_version == old_version:
            click.echo("sparkrun %s is already the latest version." % old_version)
        else:
            click.echo("sparkrun updated: %s -> %s" % (old_version, new_version))
    else:
        click.echo("sparkrun updated (could not determine new version)")


@setup.command("ssh")
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line, # comments)")
@click.option("--cluster", "cluster_name", default=None, type=CLUSTER_NAME,
              help="Use a saved cluster by name")
@click.option("--extra-hosts", default=None,
              help="Additional comma-separated hosts to include (e.g. control machine)")
@click.option("--include-self/--no-include-self", default=True, show_default=True,
              help="Include this machine's hostname in the mesh")
@click.option("--user", "-u", default=None, help="SSH username (default: current user)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")
@click.pass_context
def setup_ssh(ctx, hosts, hosts_file, cluster_name, extra_hosts, include_self, user, dry_run):
    """Set up passwordless SSH mesh across cluster hosts.

    Ensures every host can SSH to every other host without password prompts.
    Creates ed25519 keys if missing and distributes public keys.

    By default, the machine running sparkrun is included in the mesh
    (--include-self). Use --no-include-self to exclude it.

    You will be prompted for passwords on first connection to each host.

    Examples:

      sparkrun setup ssh --hosts 192.168.11.13,192.168.11.14

      sparkrun setup ssh --cluster mylab --user ubuntu

      sparkrun setup ssh --cluster mylab --extra-hosts 10.0.0.1
    """
    import os
    import subprocess

    from sparkrun.hosts import resolve_hosts
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import SparkrunConfig, get_config_root

    config = SparkrunConfig()

    # Resolve hosts and look up cluster user if applicable
    cluster_mgr = ClusterManager(get_config_root())
    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )

    # Determine the cluster's configured user (if hosts came from a cluster)
    cluster_user = None
    resolved_cluster_name = cluster_name
    if not resolved_cluster_name and not hosts and not hosts_file:
        resolved_cluster_name = cluster_mgr.get_default()
    if resolved_cluster_name:
        try:
            cluster_def = cluster_mgr.get(resolved_cluster_name)
            cluster_user = cluster_def.user
        except Exception:
            pass

    # Track original cluster hosts before extras/self are appended
    cluster_hosts = list(host_list)
    seen = set(host_list)
    added: list[str] = []
    if extra_hosts:
        for h in extra_hosts.split(","):
            h = h.strip()
            if h and h not in seen:
                host_list.append(h)
                seen.add(h)
                added.append(h)

    # Include the control machine unless opted out.
    # Use the local IP that can route to the first cluster host, since
    # remote hosts may not be able to resolve this machine's hostname.
    self_host: str | None = None
    if include_self and host_list:
        from sparkrun.orchestration.primitives import local_ip_for
        self_host = local_ip_for(host_list[0])
        if self_host and self_host not in seen:
            host_list.append(self_host)
            seen.add(self_host)
            added.append("%s (this machine)" % self_host)

    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts, --hosts-file, or --cluster.", err=True)
        sys.exit(1)

    if len(host_list) < 2:
        click.echo(
            "Error: SSH mesh requires at least 2 hosts (got %d)." % len(host_list),
            err=True,
        )
        sys.exit(1)

    # Default user: --user flag > cluster user > config ssh.user > OS user
    if user is None:
        user = cluster_user or config.ssh_user or os.environ.get("USER", "root")

    # Locate the bundled script
    from sparkrun.scripts import get_script_path
    with get_script_path("mesh_ssh_keys.sh") as script_path:
        cmd = ["bash", str(script_path), user] + host_list

        if dry_run:
            click.echo("Would run:")
            click.echo("  " + " ".join(cmd))
            return

        click.echo("Setting up SSH mesh for user '%s' across %d hosts..." % (user, len(host_list)))
        click.echo("Cluster Hosts: %s" % ", ".join(sorted(cluster_hosts)))
        if added:
            click.echo("Added: %s" % ", ".join(added))
        click.echo()

        # Run interactively — the script prompts for passwords
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line, # comments)")
@click.option("--cluster", "cluster_name", default=None, help="Use a saved cluster by name")
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def stop(ctx, recipe_name, hosts, hosts_file, cluster_name, tp_override, dry_run, config_path=None):
    """Stop a running workload.

    RECIPE_NAME identifies the recipe so the correct containers can be found.

    Examples:

      sparkrun stop glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      sparkrun stop glm-4.7-flash-awq --cluster mylab
    """
    from sparkrun.config import SparkrunConfig
    from sparkrun.hosts import resolve_hosts
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root
    from sparkrun.recipe import Recipe, find_recipe, RecipeError

    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Load recipe for cluster_id derivation
    try:
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    cluster_mgr = ClusterManager(get_config_root())
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

    # Apply TP-based host trimming to match what 'run' used for cluster_id
    host_list = _apply_tp_trimming(host_list, recipe, tp_override=tp_override)

    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers, cleanup_containers_local
    from sparkrun.orchestration.docker import generate_container_name, generate_cluster_id

    cluster_id = generate_cluster_id(recipe, host_list)
    ssh_kwargs = build_ssh_kwargs(config)

    # Build list of all possible container names for this cluster_id
    # (covers solo, Ray head/worker, and native node_N patterns)
    container_names = [
        generate_container_name(cluster_id, "solo"),
        generate_container_name(cluster_id, "head"),
        generate_container_name(cluster_id, "worker"),
    ]
    # Add per-rank node containers for native clustering
    for rank in range(len(host_list)):
        container_names.append("%s_node_%d" % (cluster_id, rank))

    is_local = len(host_list) == 1 and host_list[0] in ("localhost", "127.0.0.1", "")
    if is_local:
        cleanup_containers_local(container_names, dry_run=dry_run)
    else:
        cleanup_containers(host_list, container_names, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    click.echo("Workload stopped on %d host(s)." % len(host_list))
    sys.exit(0)


@main.command("logs")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line, # comments)")
@click.option("--cluster", "cluster_name", default=None, help="Use a saved cluster by name")
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@click.option("--tail", type=int, default=100, help="Number of log lines before following")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def logs_cmd(ctx, recipe_name, hosts, hosts_file, cluster_name, tp_override, tail, config_path=None):
    """Re-attach to logs of a running workload.

    RECIPE_NAME identifies the recipe so the correct containers can be found.

    Examples:

      sparkrun logs glm-4.7-flash-awq --hosts 192.168.11.13

      sparkrun logs glm-4.7-flash-awq --cluster mylab --tail 200
    """
    from sparkrun.bootstrap import init_sparkrun, get_runtime
    from sparkrun.config import SparkrunConfig
    from sparkrun.hosts import resolve_hosts
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root
    from sparkrun.recipe import Recipe, find_recipe, RecipeError
    from sparkrun.orchestration.docker import generate_cluster_id

    v = init_sparkrun()
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Load recipe
    try:
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Resolve hosts
    cluster_mgr = ClusterManager(get_config_root(v))
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

    # Apply TP-based host trimming to match what 'run' used for cluster_id
    host_list = _apply_tp_trimming(host_list, recipe, tp_override=tp_override)

    cluster_id = generate_cluster_id(recipe, host_list)

    # Resolve runtime so we call the correct follow_logs implementation
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    runtime.follow_logs(
        hosts=host_list,
        cluster_id=cluster_id,
        config=config,
        tail=tail,
    )


@main.group()
@click.pass_context
def cluster(ctx):
    """Manage saved cluster definitions."""
    pass


@cluster.command("create")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line)")
@click.option("-d", "--description", default="", help="Cluster description")
@click.option("--user", "-u", default=None, help="SSH username for this cluster")
@click.pass_context
def cluster_create(ctx, name, hosts, hosts_file, description, user):
    """Create a new named cluster."""
    from sparkrun.cluster_manager import ClusterManager, ClusterError
    from sparkrun.config import get_config_root
    from sparkrun.hosts import parse_hosts_file

    host_list = [h.strip() for h in hosts.split(",") if h.strip()] if hosts else []
    if hosts_file:
        host_list = parse_hosts_file(hosts_file)

    if not host_list:
        click.echo("Error: No hosts provided.", err=True)
        sys.exit(1)

    mgr = ClusterManager(get_config_root())
    try:
        mgr.create(name, host_list, description, user=user)
        click.echo(f"Cluster '{name}' created with {len(host_list)} host(s).")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("update")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line)")
@click.option("-d", "--description", default=None, help="Cluster description")
@click.option("--user", "-u", default=None, help="SSH username for this cluster")
@click.pass_context
def cluster_update(ctx, name, hosts, hosts_file, description, user):
    """Update an existing cluster."""
    from sparkrun.cluster_manager import ClusterManager, ClusterError
    from sparkrun.config import get_config_root
    from sparkrun.hosts import parse_hosts_file

    host_list = None
    if hosts:
        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    elif hosts_file:
        host_list = parse_hosts_file(hosts_file)

    if host_list is None and description is None and user is None:
        click.echo("Error: Nothing to update. Provide --hosts, --hosts-file, -d, or --user.", err=True)
        sys.exit(1)

    mgr = ClusterManager(get_config_root())
    try:
        mgr.update(name, hosts=host_list, description=description, user=user)
        click.echo(f"Cluster '{name}' updated.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("list")
@click.pass_context
def cluster_list(ctx):
    """List all saved clusters."""
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root

    mgr = ClusterManager(get_config_root())
    clusters = mgr.list_clusters()
    default_name = mgr.get_default()

    if not clusters:
        click.echo("No saved clusters.")
        return

    click.echo(f"  {'Name':<20} {'Hosts':<40} {'Description':<30}")
    click.echo("-" * 93)
    for c in clusters:
        marker = "* " if c.name == default_name else "  "
        desc = c.description or ""
        # Break hosts into lines of 2 addresses each
        host_lines = []
        for i in range(0, len(c.hosts), 2):
            host_lines.append(", ".join(c.hosts[i:i + 2]))
        first_hosts = host_lines[0] if host_lines else ""
        click.echo(f"{marker}{c.name:<20} {first_hosts:<40} {desc:<30}")
        for extra in host_lines[1:]:
            click.echo(f"  {'':<20} {extra:<40}")

    if default_name:
        click.echo(f"\n* = default cluster")


@cluster.command("show")
@click.argument("name", type=CLUSTER_NAME)
@click.pass_context
def cluster_show(ctx, name):
    """Show details of a saved cluster."""
    from sparkrun.cluster_manager import ClusterManager, ClusterError
    from sparkrun.config import get_config_root

    mgr = ClusterManager(get_config_root())
    try:
        c = mgr.get(name)
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    default_name = mgr.get_default()
    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    if c.user:
        click.echo(f"User:        {c.user}")
    click.echo(f"Default:     {'yes' if c.name == default_name else 'no'}")
    click.echo(f"Hosts ({len(c.hosts)}):")
    for h in c.hosts:
        click.echo(f"  - {h}")


@cluster.command("delete")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def cluster_delete(ctx, name, force):
    """Delete a saved cluster."""
    from sparkrun.cluster_manager import ClusterManager, ClusterError
    from sparkrun.config import get_config_root

    mgr = ClusterManager(get_config_root())

    if not force:
        click.confirm(f"Delete cluster '{name}'?", abort=True)

    try:
        mgr.delete(name)
        click.echo(f"Cluster '{name}' deleted.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("set-default")
@click.argument("name", type=CLUSTER_NAME)
@click.pass_context
def cluster_set_default(ctx, name):
    """Set the default cluster."""
    from sparkrun.cluster_manager import ClusterManager, ClusterError
    from sparkrun.config import get_config_root

    mgr = ClusterManager(get_config_root())
    try:
        mgr.set_default(name)
        click.echo(f"Default cluster set to '{name}'.")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("unset-default")
@click.pass_context
def cluster_unset_default(ctx):
    """Remove the default cluster setting."""
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root

    mgr = ClusterManager(get_config_root())
    mgr.unset_default()
    click.echo("Default cluster unset.")


@cluster.command("default")
@click.pass_context
def cluster_default(ctx):
    """Show the current default cluster."""
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root

    mgr = ClusterManager(get_config_root())
    default_name = mgr.get_default()
    if not default_name:
        click.echo("No default cluster set.")
        return

    c = mgr.get(default_name)
    click.echo(f"Name:        {c.name}")
    click.echo(f"Description: {c.description or '(none)'}")
    click.echo(f"Hosts ({len(c.hosts)}):")
    for h in c.hosts:
        click.echo(f"  - {h}")


@main.group()
@click.pass_context
def recipe(ctx):
    """Manage recipe registries and search for recipes."""
    pass


@recipe.command("list")
@click.option("--registry", default=None, help="Filter by registry name")
@click.argument("query", required=False)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_list(ctx, registry, query, config_path=None):
    """List available recipes from all registries."""
    from sparkrun.recipe import list_recipes

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    if query:
        recipes = registry_mgr.search_recipes(query)
    else:
        recipes = list_recipes(config.get_recipe_search_paths(), registry_mgr)

    if registry:
        recipes = [r for r in recipes if r.get("registry") == registry]

    if not recipes:
        click.echo("No recipes found.")
        return

    # Compute column widths from data
    reg_names = [r.get("registry", "local") for r in recipes]
    w_name = max(len("Name"), *(len(r["name"]) for r in recipes)) + 2
    w_rt = max(len("Runtime"), *(len(r["runtime"]) for r in recipes)) + 2
    w_reg = max(len("Registry"), *(len(n) for n in reg_names)) + 2
    w_file = max(len("File"), *(len(r["file"]) for r in recipes))

    click.echo(f"{'Name':<{w_name}} {'Runtime':<{w_rt}} {'Registry':<{w_reg}} {'File':<{w_file}}")
    click.echo("-" * (w_name + w_rt + w_reg + w_file + 3))
    for r, reg_name in zip(recipes, reg_names):
        click.echo(f"{r['name']:<{w_name}} {r['runtime']:<{w_rt}} {reg_name:<{w_reg}} {r['file']:<{w_file}}")


@recipe.command("search")
@click.argument("query")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_search(ctx, query, config_path=None, ):
    """Search for recipes by name, model, or description."""
    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    recipes = registry_mgr.search_recipes(query)

    if not recipes:
        click.echo(f"No recipes found matching '{query}'.")
        return

    # Compute column widths from data
    models = [r.get("model", "") for r in recipes]
    reg_names = [r.get("registry", "local") for r in recipes]
    w_name = max(len("Name"), *(len(r["name"]) for r in recipes)) + 2
    w_rt = max(len("Runtime"), *(len(r["runtime"]) for r in recipes)) + 2
    w_model = max(len("Model"), *(len(m) for m in models)) + 2
    w_reg = max(len("Registry"), *(len(n) for n in reg_names))

    click.echo(f"{'Name':<{w_name}} {'Runtime':<{w_rt}} {'Model':<{w_model}} {'Registry':<{w_reg}}")
    click.echo("-" * (w_name + w_rt + w_model + w_reg + 3))
    for r, model, reg_name in zip(recipes, models, reg_names):
        click.echo(f"{r['name']:<{w_name}} {r['runtime']:<{w_rt}} {model:<{w_model}} {reg_name:<{w_reg}}")


@recipe.command("show")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_show(ctx, recipe_name, no_vram, config_path=None, ):
    """Show detailed recipe information."""
    from sparkrun.recipe import Recipe, find_recipe, RecipeError

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    try:
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    reg_name = registry_mgr.registry_for_path(recipe_path) if registry_mgr else None
    _display_recipe_detail(recipe, show_vram=not no_vram, registry_name=reg_name)


@recipe.command("validate")
@click.argument("recipe_name", type=RECIPE_NAME)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_validate(ctx, recipe_name, config_path=None):
    """Validate a recipe file."""
    from sparkrun.bootstrap import init_sparkrun, get_runtime
    from sparkrun.recipe import Recipe, find_recipe, RecipeError

    v = init_sparkrun()
    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    try:
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    issues = recipe.validate()

    try:
        runtime = get_runtime(recipe.runtime, v)
        issues.extend(runtime.validate_recipe(recipe))
    except ValueError:
        issues.append(f"Unknown runtime: {recipe.runtime}")

    if issues:
        click.echo(f"Recipe '{recipe.name}' has {len(issues)} issue(s):")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)
    else:
        click.echo(f"Recipe '{recipe.name}' is valid.")


@recipe.command("vram")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--max-model-len", type=int, default=None, help="Override max sequence length")
@click.option("--gpu-mem", type=float, default=None,
              help="Override gpu_memory_utilization (0.0-1.0)")
@click.option("--no-auto-detect", is_flag=True, help="Skip HuggingFace model auto-detection")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_vram(ctx, recipe_name, tensor_parallel, max_model_len, gpu_mem, no_auto_detect, config_path=None):
    """Estimate VRAM usage for a recipe on DGX Spark.

    Shows model weight size, KV cache requirements, GPU memory budget,
    and whether the configuration fits within DGX Spark memory.

    Examples:

      sparkrun recipe vram glm-4.7-flash-awq

      sparkrun recipe vram glm-4.7-flash-awq --tp 2

      sparkrun recipe vram my-recipe.yaml --max-model-len 8192 --gpu-mem 0.9
    """
    from sparkrun.recipe import Recipe, find_recipe, RecipeError

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    try:
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Recipe:  {recipe.name}")
    click.echo(f"Model:   {recipe.model}")
    click.echo(f"Runtime: {recipe.runtime}")

    cli_overrides = {}
    if tensor_parallel is not None:
        cli_overrides["tensor_parallel"] = tensor_parallel
    if max_model_len is not None:
        cli_overrides["max_model_len"] = max_model_len
    if gpu_mem is not None:
        cli_overrides["gpu_memory_utilization"] = gpu_mem

    _display_vram_estimate(recipe, cli_overrides=cli_overrides, auto_detect=not no_auto_detect)


@recipe.command("update")
@click.option("--registry", default=None, help="Update specific registry")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_update(ctx, registry, config_path=None, ):
    """Update recipe registries from git."""
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.update(registry)
        if registry:
            click.echo(f"Registry '{registry}' updated successfully.")
        else:
            click.echo("All registries updated successfully.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@recipe.command("registries")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_registries(ctx, config_path=None, ):
    """List configured recipe registries."""
    config, registry_mgr = _get_config_and_registry(config_path)

    registries = registry_mgr.list_registries()

    if not registries:
        click.echo("No registries configured.")
        return

    # Table header
    click.echo(f"{'Name':<20} {'URL':<40} {'Subpath':<20} {'Enabled':<8}")
    click.echo("-" * 88)
    for reg in registries:
        url = reg.url[:38] + ".." if len(reg.url) > 40 else reg.url
        enabled = "yes" if reg.enabled else "no"
        click.echo(f"{reg.name:<20} {url:<40} {reg.subpath:<20} {enabled:<8}")


@recipe.command("add-registry")
@click.argument("name")
@click.option("--url", required=True, help="Git repository URL")
@click.option("--subpath", required=True, help="Path to recipes within repo")
@click.option("-d", "--description", default="", help="Registry description")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_add_registry(ctx, name, url, subpath, description, config_path=None, ):
    """Add a new recipe registry."""
    from sparkrun.registry import RegistryEntry, RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        entry = RegistryEntry(
            name=name,
            url=url,
            subpath=subpath,
            description=description,
            enabled=True,
        )
        registry_mgr.add_registry(entry)
        click.echo(f"Registry '{name}' added successfully.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@recipe.command("remove-registry")
@click.argument("name")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_remove_registry(ctx, name, config_path=None, ):
    """Remove a recipe registry."""
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.remove_registry(name)
        click.echo(f"Registry '{name}' removed successfully.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _distribute_resources(
        image: str,
        model: str,
        host_list: list[str],
        cache_dir: str,
        config,
        dry_run: bool,
        skip_ib: bool,
        model_revision: str | None = None,
) -> dict[str, str] | None:
    """Detect IB, distribute container image and model to target hosts.

    Performs InfiniBand detection (for both NCCL env and IB transfer IPs),
    then distributes the container image and model from local to all
    remote hosts using the fast IB network when available.

    For localhost targets, only ensures the image/model exist locally.

    Args:
        image: Container image reference.
        model: HuggingFace model identifier (may be empty).
        host_list: Target hostnames/IPs.
        cache_dir: HuggingFace cache directory.
        config: SparkrunConfig instance.
        dry_run: Show what would be done without executing.
        skip_ib: Skip InfiniBand detection.
        model_revision: Optional HuggingFace model revision to pin.

    Returns:
        Pre-detected NCCL environment dict, or ``None`` if IB detection
        was skipped or not applicable (localhost).  The caller should
        pass this to ``runtime.run(nccl_env=...)`` to avoid redundant
        IB detection.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.infiniband import detect_ib_for_hosts
    from sparkrun.containers.distribute import distribute_image_from_local
    from sparkrun.containers.registry import ensure_image
    from sparkrun.models.distribute import distribute_model_from_local
    from sparkrun.models.download import download_model

    is_local = (
            len(host_list) <= 1
            and host_list[0] in ("localhost", "127.0.0.1", "")
    )

    if is_local:
        # Local-only: just ensure image and model exist, no SSH needed
        click.echo("Ensuring container image is available locally...")
        ensure_image(image, dry_run=dry_run)
        if model:
            click.echo(f"Ensuring model {model} is available locally...")
            download_model(model, cache_dir=cache_dir, revision=model_revision, dry_run=dry_run)
        return None  # let runtime handle its own local IB detection

    ssh_kwargs = build_ssh_kwargs(config)
    nccl_env: dict[str, str] = {}
    transfer_hosts: list[str] | None = None

    # Step 1: Detect InfiniBand for NCCL env + transfer routing
    if not skip_ib:
        # click.echo(f"Detecting InfiniBand on {len(host_list)} host(s)...")
        ib_result = detect_ib_for_hosts(
            host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        nccl_env = ib_result.nccl_env
        if ib_result.ib_ip_map:
            transfer_hosts = [
                ib_result.ib_ip_map.get(h, h) for h in host_list
            ]
            click.echo(
                f"  Using IB network for transfers "
                f"({len(ib_result.ib_ip_map)}/{len(host_list)} hosts)"
            )

    # Step 2: Distribute container image
    # click.echo(f"Distributing image to {len(host_list)} host(s)...")
    img_failed = distribute_image_from_local(
        image, host_list,
        transfer_hosts=transfer_hosts,
        dry_run=dry_run, **ssh_kwargs,
    )
    if img_failed:
        click.echo(
            "Warning: Image distribution failed on: %s" % ", ".join(img_failed),
            err=True,
        )

    # Step 3: Distribute model
    if model:
        click.echo(f"Distributing model {model} to {len(host_list)} host(s)...")
        mdl_failed = distribute_model_from_local(
            model, host_list,
            cache_dir=cache_dir,
            revision=model_revision,
            transfer_hosts=transfer_hosts,
            dry_run=dry_run, **ssh_kwargs,
        )
        if mdl_failed:
            click.echo(
                "Warning: Model distribution failed on: %s" % ", ".join(mdl_failed),
                err=True,
            )

    click.echo("Distribution complete.")
    click.echo()
    return nccl_env
