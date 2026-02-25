"""sparkrun recipe group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _display_recipe_detail,
    _display_vram_estimate,
    _get_config_and_registry,
    _load_recipe,
)


@click.group()
@click.pass_context
def recipe(ctx):
    """Manage recipe registries and search for recipes."""
    pass


@recipe.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query", required=False)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_list(ctx, registry, runtime, query, config_path=None):
    """List available recipes from all registries."""
    from sparkrun.recipe import list_recipes, filter_recipes
    from sparkrun.utils.cli_formatters import format_recipe_table

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    if query:
        recipes = registry_mgr.search_recipes(query)
    else:
        recipes = list_recipes(config.get_recipe_search_paths(), registry_mgr)

    recipes = filter_recipes(recipes, runtime=runtime, registry=registry)
    click.echo(format_recipe_table(recipes, show_file=True))


@recipe.command("search")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_search(ctx, registry, runtime, query, config_path=None):
    """Search for recipes by name, model, or description."""
    from sparkrun.recipe import filter_recipes
    from sparkrun.utils.cli_formatters import format_recipe_table

    config, registry_mgr = _get_config_and_registry(config_path)
    registry_mgr.ensure_initialized()

    recipes = registry_mgr.search_recipes(query)
    recipes = filter_recipes(recipes, runtime=runtime, registry=registry)

    if not recipes:
        click.echo(f"No recipes found matching '{query}'.")
        return

    click.echo(format_recipe_table(recipes, show_model=True))


@recipe.command("show")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_show(ctx, recipe_name, no_vram, tensor_parallel, config_path=None):
    """Show detailed recipe information."""
    config, _ = _get_config_and_registry(config_path)
    recipe, recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    cli_overrides = {}
    if tensor_parallel is not None:
        cli_overrides["tensor_parallel"] = tensor_parallel

    reg_name = registry_mgr.registry_for_path(recipe_path) if registry_mgr else None
    _display_recipe_detail(recipe, show_vram=not no_vram, registry_name=reg_name,
                           cli_overrides=cli_overrides or None)


@recipe.command("validate")
@click.argument("recipe_name", type=RECIPE_NAME)
# @click.option("--config", "config_path", default=None, help="Path to config file")
@click.pass_context
def recipe_validate(ctx, recipe_name, config_path=None):
    """Validate a recipe file."""
    from sparkrun.bootstrap import init_sparkrun, get_runtime

    v = init_sparkrun()
    config, _ = _get_config_and_registry(config_path)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

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
    config, _ = _get_config_and_registry(config_path)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

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
        # Count how many registries will be updated
        if registry:
            entry = registry_mgr.get_registry(registry)
            if not getattr(entry, "enabled", True):
                click.echo(
                    f"Error: Registry '{registry}' is disabled; enable it in the config before updating.",
                    err=True,
                )
                sys.exit(1)
            entries = [entry]
        else:
            entries = [e for e in registry_mgr.list_registries() if e.enabled]

        count = len(entries)
        if count == 0:
            click.echo("No enabled registries to update.")
            return

        click.echo(f"Updating {count} registr{'y' if count == 1 else 'ies'}...")

        def _progress(name: str, success: bool) -> None:
            status = "done" if success else "FAILED"
            click.echo(f"  Updating {name}... {status}")

        results = registry_mgr.update(registry, progress=_progress)
        succeeded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        if failed:
            click.echo(f"{succeeded} of {count} registries updated ({failed} failed).")
        else:
            click.echo(f"{succeeded} registr{'y' if succeeded == 1 else 'ies'} updated.")
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
