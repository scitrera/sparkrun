"""sparkrun stop and logs commands."""

from __future__ import annotations

import sys

import click

from ._common import (
    RECIPE_NAME,
    _apply_tp_trimming,
    _load_recipe,
    _resolve_hosts_or_exit,
    _setup_logging,
    dry_run_option,
    host_options,
)


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@click.option("--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)")
@dry_run_option
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
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Load recipe for cluster_id derivation
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)

    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts or configure defaults.", err=True)
        sys.exit(1)

    # Apply TP-based host trimming to match what 'run' used for cluster_id
    host_list = _apply_tp_trimming(host_list, recipe, tp_override=tp_override)

    from sparkrun.orchestration.primitives import build_ssh_kwargs, cleanup_containers, cleanup_containers_local
    from sparkrun.orchestration.docker import enumerate_cluster_containers
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    cluster_id = generate_cluster_id(recipe, host_list)
    ssh_kwargs = build_ssh_kwargs(config)

    container_names = enumerate_cluster_containers(cluster_id, len(host_list))

    from sparkrun.hosts import is_local_host
    is_local = len(host_list) == 1 and is_local_host(host_list[0])
    if is_local:
        cleanup_containers_local(container_names, dry_run=dry_run)
    else:
        cleanup_containers(host_list, container_names, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    click.echo("Workload stopped on %d host(s)." % len(host_list))
    sys.exit(0)


@click.command("logs")
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
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
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    v = init_sparkrun()
    _setup_logging(ctx.obj["verbose"])
    config = SparkrunConfig(config_path) if config_path else SparkrunConfig()

    # Load recipe
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)

    # Resolve hosts
    host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config, v)

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
