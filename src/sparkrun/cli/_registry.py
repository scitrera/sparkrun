"""sparkrun registry group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    REGISTRY_NAME,
    _get_config_and_registry,
)


@click.group()
@click.pass_context
def registry(ctx):
    """Manage recipe registries."""
    pass


@registry.command("list")
@click.option("--show-disabled", is_flag=True, help="Also show disabled registries")
@click.option("--only-show-visible", is_flag=True, help="Only show visible registries")
@click.pass_context
def registry_list(ctx, show_disabled, only_show_visible, config_path=None):
    """List configured recipe registries.

    By default, shows all enabled registries (including hidden ones).
    """
    config, registry_mgr = _get_config_and_registry(config_path)
    registries = registry_mgr.list_registries()

    if not registries:
        click.echo("No registries configured.")
        return

    # Default: all enabled registries
    display_registries = registries
    if not show_disabled:
        display_registries = [r for r in display_registries if r.enabled]
    if only_show_visible:
        display_registries = [r for r in display_registries if r.visible]

    if not display_registries:
        click.echo("No matching registries found.")
        return

    # Determine which content columns to show
    has_tuning = any(r.tuning_subpath for r in display_registries)
    has_benchmarks = any(r.benchmark_subpath for r in display_registries)

    # Table header
    header = f"{'Name':<25} {'URL':<45} {'Enabled':<9} {'Visible':<9}"
    sep_width = 88
    if has_tuning:
        header += f" {'Tuning':<8}"
        sep_width += 9
    if has_benchmarks:
        header += f" {'Bench':<7}"
        sep_width += 8
    click.echo(header)
    click.echo("-" * sep_width)

    for reg in display_registries:
        url = reg.url[:43] + ".." if len(reg.url) > 45 else reg.url
        enabled = "yes" if reg.enabled else "no"
        visible = "yes" if reg.visible else "no"
        row = f"{reg.name:<25} {url:<45} {enabled:<9} {visible:<9}"
        if has_tuning:
            tuning = "yes" if reg.tuning_subpath else "no"
            row += f" {tuning:<8}"
        if has_benchmarks:
            bench = "yes" if reg.benchmark_subpath else "no"
            row += f" {bench:<7}"
        click.echo(row)


# @registry.command("manual-add")
# @click.argument("name")
# @click.option("--url", required=True, help="Git repository URL")
# @click.option("--subpath", required=True, help="Path to recipes within repo")
# @click.option("-d", "--description", default="", help="Registry description")
# @click.option("--visible/--hidden", default=True, help="Registry visibility in default listings")
# @click.option("--tuning-subpath", default="", help="Path to tuning configs within repo")
# @click.option("--benchmark-subpath", default="", help="Path to benchmark profiles within repo")
# @click.pass_context
# def registry_add(ctx, name, url, subpath, description, visible, tuning_subpath, benchmark_subpath, config_path=None):
#     """Add a new recipe registry."""
#     from sparkrun.registry import RegistryEntry, RegistryError
#
#     config, registry_mgr = _get_config_and_registry(config_path)
#
#     try:
#         entry = RegistryEntry(
#             name=name,
#             url=url,
#             subpath=subpath,
#             description=description,
#             enabled=True,
#             visible=visible,
#             tuning_subpath=tuning_subpath,
#             benchmark_subpath=benchmark_subpath,
#         )
#         registry_mgr.add_registry(entry)
#         click.echo(f"Registry '{name}' added successfully.")
#     except RegistryError as e:
#         click.echo(f"Error: {e}", err=True)
#         sys.exit(1)


@registry.command("add")
@click.argument("url")
@click.pass_context
def registry_add_url(ctx, url, config_path=None):
    """Add registries from a repository's .sparkrun/registry.yaml manifest.

    Clones the repository, reads the manifest, and adds all declared registries.

    Examples:

      sparkrun registry add https://github.com/spark-arena/recipe-registry
    """
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        click.echo("Discovering registries from %s..." % url)
        added = registry_mgr.add_registry_from_url(url)
        if added:
            click.echo("Added %d registr%s:" % (len(added), "y" if len(added) == 1 else "ies"))
            for entry in added:
                vis = "" if entry.visible else " (hidden)"
                click.echo("  %s — %s%s" % (entry.name, entry.description, vis))
        else:
            click.echo("No new registries added (all may already exist).")
    except RegistryError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)


@registry.command("remove")
@click.argument("name", type=REGISTRY_NAME)
@click.pass_context
def registry_remove(ctx, name, config_path=None):
    """Remove a recipe registry."""
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.remove_registry(name)
        click.echo(f"Registry '{name}' removed successfully.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@registry.command("enable")
@click.argument("name", type=REGISTRY_NAME)
@click.pass_context
def registry_enable(ctx, name, config_path=None):
    """Enable a disabled registry."""
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.enable_registry(name)
        click.echo(f"Registry '{name}' enabled.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@registry.command("disable")
@click.argument("name", type=REGISTRY_NAME)
@click.pass_context
def registry_disable(ctx, name, config_path=None):
    """Disable a registry (recipes will not appear in searches)."""
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        registry_mgr.disable_registry(name)
        click.echo(f"Registry '{name}' disabled.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@registry.command("revert-to-default")
@click.option("--update", "run_update", is_flag=True, help="Also run registry update after reset")
@click.pass_context
def registry_revert_to_default(ctx, run_update, config_path=None):
    """Reset registries to defaults (deletes config and re-initializes).

    Removes the current registries.yaml and re-discovers registries from
    the default manifest URLs.  If discovery fails (offline, etc.), falls
    back to hardcoded defaults.

    Examples:

      sparkrun registry revert-to-default

      sparkrun registry revert-to-default --update
    """
    config, registry_mgr = _get_config_and_registry(config_path)

    entries = registry_mgr.reset_to_defaults()
    click.echo("Registries reset to defaults (%d entries):" % len(entries))
    for entry in entries:
        vis = "" if entry.visible else " (hidden)"
        click.echo("  %s — %s%s" % (entry.name, entry.description or entry.url, vis))

    if run_update:
        click.echo()
        ctx.invoke(registry_update)


@registry.command("update")
@click.argument("name", required=False, default=None, type=REGISTRY_NAME)
@click.pass_context
def registry_update(ctx, name, config_path=None):
    """Update recipe registries from git.

    If NAME is given, update only that registry. Otherwise update all enabled registries.
    """
    from sparkrun.registry import RegistryError

    config, registry_mgr = _get_config_and_registry(config_path)

    try:
        if name:
            entry = registry_mgr.get_registry(name)
            if not entry.enabled:
                click.echo(
                    f"Error: Registry '{name}' is disabled; enable it before updating.",
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

        def _progress(prog_name: str, success: bool) -> None:
            status = "done" if success else "FAILED"
            click.echo(f"  Updating {prog_name}... {status}")

        results = registry_mgr.update(name, progress=_progress)
        succeeded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)

        if failed:
            click.echo(f"{succeeded} of {count} registries updated ({failed} failed).")
        else:
            click.echo(f"{succeeded} registr{'y' if succeeded == 1 else 'ies'} updated.")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
