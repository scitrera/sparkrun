"""sparkrun CLI — launch inference workloads on DGX Spark."""

from __future__ import annotations

import logging
import sys

import click

from sparkrun import __version__


def _get_config_and_registry(config_path=None):
    """Create SparkrunConfig and RegistryManager."""
    from sparkrun.config import SparkrunConfig
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    registry_mgr = config.get_registry_manager()
    return config, registry_mgr


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
@click.option("--cluster-id", default="sparkrun0", help="Cluster identifier for container naming")
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None, help="Override GPU memory utilization")
@click.option("--image", default=None, help="Override container image")
@click.option("--cache-dir", default=None, help="HuggingFace cache directory")
@click.option("--ray-port", type=int, default=46379, help="Ray GCS port")
@click.option("--dashboard", is_flag=True, help="Enable Ray dashboard on head node")
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port")
@click.option("--setup", is_flag=True, help="Pull image and download model before running")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--skip-ib", is_flag=True, help="Skip InfiniBand detection")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def run(ctx, recipe_name, hosts, hosts_file, cluster_name, solo, cluster_id, port, tensor_parallel,
        gpu_mem, image, cache_dir, ray_port, dashboard, dashboard_port, setup,
        dry_run, foreground, skip_ib, config_path, extra_args):
    """Run an inference recipe.

    RECIPE_NAME can be a recipe file path or a name to search for.

    Examples:

      sparkrun run glm-4.7-flash-awq --solo

      sparkrun run glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      sparkrun run glm-4.7-flash-awq --cluster mylab

      sparkrun run my-recipe.yaml --port 9000 --gpu-mem 0.8
    """
    from sparkrun.bootstrap import init_sparkrun, get_runtime
    from sparkrun.recipe import Recipe, find_recipe, RecipeError
    from sparkrun.config import SparkrunConfig

    v = init_sparkrun()
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

    # Build overrides from CLI options
    overrides = {}
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

    # Determine mode
    is_solo = solo or len(host_list) <= 1
    if recipe.mode == "cluster" and is_solo and not solo:
        click.echo("Warning: Recipe requires cluster mode but only one host specified", err=True)
    if recipe.mode == "solo":
        is_solo = True

    # Handle delegating runtimes (eugr-vllm)
    if runtime.is_delegating_runtime():
        from sparkrun.runtimes.eugr_vllm import EugrVllmRuntime
        if isinstance(runtime, EugrVllmRuntime):
            rc = runtime.run_delegated(
                recipe=recipe,
                overrides=overrides,
                hosts=host_list if not is_solo else None,
                solo=is_solo,
                setup=setup,
                dry_run=dry_run,
                cache_dir=config.cache_dir,
            )
            sys.exit(rc)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    # Setup phase (pull image + download model)
    if setup:
        _run_setup(container_image, recipe.model, host_list, cache_dir, config, dry_run)

    # Generate serve command
    serve_command = runtime.generate_command(
        recipe=recipe,
        overrides=overrides,
        is_cluster=not is_solo,
        num_nodes=len(host_list),
        head_ip=None,  # will be determined during cluster launch
    )

    click.echo(f"Runtime:   {runtime.runtime_name}")
    click.echo(f"Image:     {container_image}")
    click.echo(f"Model:     {recipe.model}")
    click.echo(f"Mode:      {'solo' if is_solo else f'cluster ({len(host_list)} nodes)'}")

    # Show VRAM estimate inline
    _display_vram_estimate(recipe, cli_overrides=overrides, auto_detect=True)

    # Show cluster/host info
    click.echo()
    click.echo(f"Hosts:     {host_source}")
    if is_solo:
        target = host_list[0] if host_list else "localhost"
        click.echo(f"  Target:  {target}")
    else:
        click.echo(f"  Head:    {host_list[0]}")
        if len(host_list) > 1:
            click.echo(f"  Workers: {', '.join(host_list[1:])}")

    # Show full serve command
    click.echo()
    click.echo("Serve command:")
    for line in serve_command.strip().splitlines():
        click.echo(f"  {line}")
    click.echo()

    # Launch
    if is_solo:
        from sparkrun.orchestration.solo import run_solo
        target_host = host_list[0] if host_list else "localhost"
        rc = run_solo(
            host=target_host,
            image=container_image,
            serve_command=serve_command,
            cluster_id=cluster_id,
            env=recipe.env,
            cache_dir=cache_dir or str(config.hf_cache_dir),
            config=config,
            dry_run=dry_run,
            detached=not foreground,
            skip_ib_detect=skip_ib,
        )
    else:
        from sparkrun.orchestration.cluster import run_cluster
        runtime_env = runtime.get_cluster_env(
            head_ip="<pending>", num_nodes=len(host_list)
        )
        rc = run_cluster(
            hosts=host_list,
            image=container_image,
            serve_command=serve_command,
            cluster_id=cluster_id,
            ray_port=ray_port,
            dashboard_port=dashboard_port,
            dashboard=dashboard,
            env=recipe.env,
            runtime_cluster_env=runtime_env,
            cache_dir=cache_dir or str(config.hf_cache_dir),
            config=config,
            dry_run=dry_run,
            detached=not foreground,
            skip_ib_detect=skip_ib,
        )

    sys.exit(rc)


@main.command("list")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def list_cmd(ctx, config_path):
    """List available recipes."""
    from sparkrun.recipe import list_recipes
    from sparkrun.config import SparkrunConfig

    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    registry_mgr = config.get_registry_manager()
    registry_mgr.ensure_initialized()
    recipes = list_recipes(config.get_recipe_search_paths(), registry_mgr)

    if not recipes:
        click.echo("No recipes found.")
        return

    # Table header
    click.echo(f"{'Name':<35} {'Runtime':<12} {'File':<30}")
    click.echo("-" * 77)
    for r in recipes:
        click.echo(f"{r['name']:<35} {r['runtime']:<12} {r['file']:<30}")


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
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def show(ctx, recipe_name, no_vram, config_path):
    """Show detailed recipe information."""
    from sparkrun.recipe import Recipe, find_recipe, RecipeError
    from sparkrun.config import SparkrunConfig

    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    registry_mgr = config.get_registry_manager()
    registry_mgr.ensure_initialized()

    try:
        recipe_path = find_recipe(recipe_name, config.get_recipe_search_paths(), registry_mgr)
        recipe = Recipe.load(recipe_path)
    except RecipeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    reg_name = registry_mgr.registry_for_path(recipe_path) if registry_mgr else None
    _display_recipe_detail(recipe, show_vram=not no_vram, registry_name=reg_name)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.option("--max-model-len", type=int, default=None, help="Override max sequence length")
@click.option("--gpu-mem", type=float, default=None,
              help="Override gpu_memory_utilization (0.0-1.0)")
@click.option("--no-auto-detect", is_flag=True, help="Skip HuggingFace model auto-detection")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def vram(ctx, recipe_name, tensor_parallel, max_model_len, gpu_mem, no_auto_detect, config_path):
    """Estimate VRAM usage for a recipe on DGX Spark.

    Shows model weight size, KV cache requirements, GPU memory budget,
    and whether the configuration fits within DGX Spark memory.

    Examples:

      sparkrun vram glm-4.7-flash-awq

      sparkrun vram glm-4.7-flash-awq --tp 2

      sparkrun vram my-recipe.yaml --max-model-len 8192 --gpu-mem 0.9
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


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def validate(ctx, recipe_name, config_path):
    """Validate a recipe file."""
    from sparkrun.bootstrap import init_sparkrun, get_runtime
    from sparkrun.recipe import Recipe, find_recipe, RecipeError
    from sparkrun.config import SparkrunConfig

    v = init_sparkrun()
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
    registry_mgr = config.get_registry_manager()
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


@main.command()
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line, # comments)")
@click.option("--cluster", "cluster_name", default=None, help="Use a saved cluster by name")
@click.option("--cluster-id", default="sparkrun0", help="Cluster identifier")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be done")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def stop(ctx, hosts, hosts_file, cluster_name, cluster_id, dry_run, config_path):
    """Stop a running workload.

    Examples:

      sparkrun stop --hosts 192.168.11.13,192.168.11.14

      sparkrun stop --cluster mylab
    """
    from sparkrun.config import SparkrunConfig
    from sparkrun.hosts import resolve_hosts
    from sparkrun.cluster_manager import ClusterManager
    from sparkrun.config import get_config_root

    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()
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

    if len(host_list) == 1:
        from sparkrun.orchestration.solo import stop_solo
        rc = stop_solo(host_list[0], cluster_id, config, dry_run)
    else:
        from sparkrun.orchestration.cluster import stop_cluster
        rc = stop_cluster(host_list, cluster_id, config, dry_run)

    if rc == 0:
        click.echo("Workload stopped.")
    sys.exit(rc)


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
@click.pass_context
def cluster_create(ctx, name, hosts, hosts_file, description):
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
        mgr.create(name, host_list, description)
        click.echo(f"Cluster '{name}' created with {len(host_list)} host(s).")
    except ClusterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cluster.command("update")
@click.argument("name", type=CLUSTER_NAME)
@click.option("--hosts", "-H", default=None, help="Comma-separated host list")
@click.option("--hosts-file", default=None, help="File with hosts (one per line)")
@click.option("-d", "--description", default=None, help="Cluster description")
@click.pass_context
def cluster_update(ctx, name, hosts, hosts_file, description):
    """Update an existing cluster."""
    from sparkrun.cluster_manager import ClusterManager, ClusterError
    from sparkrun.config import get_config_root
    from sparkrun.hosts import parse_hosts_file

    host_list = None
    if hosts:
        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    elif hosts_file:
        host_list = parse_hosts_file(hosts_file)

    if host_list is None and description is None:
        click.echo("Error: Nothing to update. Provide --hosts, --hosts-file, or -d.", err=True)
        sys.exit(1)

    mgr = ClusterManager(get_config_root())
    try:
        mgr.update(name, hosts=host_list, description=description)
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
        hosts_str = ", ".join(c.hosts)
        if len(hosts_str) > 37:
            hosts_str = hosts_str[:37] + "..."
        click.echo(f"{marker}{c.name:<20} {hosts_str:<40} {c.description:<30}")

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
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_list(ctx, registry, query, config_path):
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

    # Table header
    click.echo(f"{'Name':<30} {'Runtime':<12} {'Registry':<20} {'File':<25}")
    click.echo("-" * 87)
    for r in recipes:
        reg_name = r.get("registry", "local")
        click.echo(f"{r['name']:<30} {r['runtime']:<12} {reg_name:<20} {r['file']:<25}")


@recipe.command("search")
@click.argument("query")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_search(ctx, query, config_path):
    """Search for recipes by name, model, or description."""
    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    recipes = registry_mgr.search_recipes(query)

    if not recipes:
        click.echo(f"No recipes found matching '{query}'.")
        return

    # Table header
    click.echo(f"{'Name':<30} {'Runtime':<12} {'Model':<30} {'Registry':<20}")
    click.echo("-" * 92)
    for r in recipes:
        model = r.get("model", "")[:28] + ".." if len(r.get("model", "")) > 30 else r.get("model", "")
        reg_name = r.get("registry", "local")
        click.echo(f"{r['name']:<30} {r['runtime']:<12} {model:<30} {reg_name:<20}")


@recipe.command("show")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_show(ctx, recipe_name, no_vram, config_path):
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


@recipe.command("update")
@click.option("--registry", default=None, help="Update specific registry")
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_update(ctx, registry, config_path):
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
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_registries(ctx, config_path):
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
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_add_registry(ctx, name, url, subpath, description, config_path):
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
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_remove_registry(ctx, name, config_path):
    """Remove a recipe registry."""
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.remove_registry(name)
        click.echo(f"Registry '{name}' removed successfully.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _run_setup(image: str, model: str, hosts: list[str],
               cache_dir: str | None, config, dry_run: bool):
    """Pull image and download model."""
    from sparkrun.containers.registry import ensure_image
    from sparkrun.containers.sync import sync_image_to_hosts
    from sparkrun.models.download import download_model
    from sparkrun.models.sync import sync_model_to_hosts

    click.echo("Setup: Ensuring container image is available...")
    ensure_image(image, dry_run=dry_run)

    if hosts and len(hosts) > 1:
        click.echo(f"Setup: Syncing image to {len(hosts)} hosts...")
        sync_image_to_hosts(
            image, hosts,
            ssh_user=config.ssh_user,
            ssh_key=config.ssh_key,
            dry_run=dry_run,
        )

    if model:
        click.echo(f"Setup: Downloading model {model}...")
        download_model(model, cache_dir=cache_dir, dry_run=dry_run)

        if hosts and len(hosts) > 1:
            click.echo(f"Setup: Syncing model to {len(hosts)} hosts...")
            sync_model_to_hosts(
                model, hosts,
                cache_dir=cache_dir,
                ssh_user=config.ssh_user,
                ssh_key=config.ssh_key,
                dry_run=dry_run,
            )

    click.echo("Setup complete.")
    click.echo()
