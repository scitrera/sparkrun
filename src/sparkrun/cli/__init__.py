"""sparkrun CLI — launch inference workloads on DGX Spark."""

from __future__ import annotations

import click

from sparkrun import __version__
from ._common import (
    RECIPE_NAME,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _setup_logging,
    dry_run_option,
    host_options,
)
from ._cluster import cluster, cluster_status
from ._recipe import recipe, recipe_list, recipe_search, recipe_show
from ._run import run
from ._setup import setup
from ._stop_logs import logs_cmd, stop
from ._tune import tune


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose/debug output")
@click.version_option(__version__, prog_name="sparkrun")
@click.pass_context
def main(ctx, verbose):
    """sparkrun — Launch inference workloads on NVIDIA DGX Spark systems."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# Register command groups and commands
main.add_command(run)
main.add_command(stop)
main.add_command(logs_cmd)
main.add_command(setup)
main.add_command(tune)
main.add_command(cluster)
main.add_command(recipe)


# ---------------------------------------------------------------------------
# Top-level aliases
# ---------------------------------------------------------------------------

@main.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query", required=False)
@click.pass_context
def list_cmd(ctx, registry, runtime, query):
    """List available recipes (alias for 'recipe list')."""
    ctx.invoke(recipe_list, registry=registry, runtime=runtime, query=query)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None,
              help="Override tensor parallelism")
@click.pass_context
def show(ctx, recipe_name, no_vram, tensor_parallel):
    """Show detailed recipe information (alias for 'recipe show')."""
    ctx.invoke(recipe_show, recipe_name=recipe_name, no_vram=no_vram, tensor_parallel=tensor_parallel)


@main.command("search")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query")
@click.pass_context
def search_cmd(ctx, registry, runtime, query):
    """Search for recipes by name, model, or description (alias for 'recipe search')."""
    ctx.invoke(recipe_search, registry=registry, runtime=runtime, query=query)


@main.command("status")
@host_options
@dry_run_option
@click.pass_context
def status(ctx, hosts, hosts_file, cluster_name, dry_run):
    """Show sparkrun containers running on cluster hosts (alias for 'cluster status')."""
    ctx.invoke(cluster_status, hosts=hosts, hosts_file=hosts_file,
               cluster_name=cluster_name, dry_run=dry_run)
