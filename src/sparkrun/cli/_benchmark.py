"""sparkrun benchmark commands — list and show benchmark profiles."""

from __future__ import annotations

import sys

import click
import yaml

from ._common import (
    PROFILE_NAME,
    _get_config_and_registry,
)


@click.group()
def benchmark():
    """Manage benchmark profiles."""
    pass


@benchmark.command("list-profiles")
@click.option("--all", "-a", "show_all", is_flag=True, default=False,
              help="Include profiles from hidden registries")
@click.option("--registry", default=None, help="Filter by registry name")
def list_profiles(show_all, registry):
    """List available benchmark profiles across registries."""
    _config, registry_mgr = _get_config_and_registry()

    profiles = registry_mgr.list_benchmark_profiles(
        registry_name=registry, include_hidden=show_all,
    )

    if not profiles:
        if registry:
            click.echo("No benchmark profiles found in registry '%s'." % registry)
        else:
            click.echo("No benchmark profiles found.")
        return

    # Calculate column widths
    name_width = max(len(p["file"]) for p in profiles)
    reg_width = max(len(p["registry"]) for p in profiles)
    name_width = max(name_width, 4)  # min header width
    reg_width = max(reg_width, 8)

    # Print header
    header = "%-*s  %-*s  %s" % (name_width, "Name", reg_width, "Registry", "Description")
    click.echo(header)
    click.echo("-" * len(header))

    for p in profiles:
        desc = p.get("description", "")
        # Truncate long descriptions
        if len(desc) > 60:
            desc = desc[:57] + "..."
        click.echo("%-*s  %-*s  %s" % (name_width, p["file"], reg_width, p["registry"], desc))


@benchmark.command("show-profile")
@click.argument("profile_name", type=PROFILE_NAME)
def show_profile(profile_name):
    """Show detailed benchmark profile information."""
    from sparkrun.recipe import find_benchmark_profile, ProfileError, ProfileAmbiguousError

    config, registry_mgr = _get_config_and_registry()

    try:
        profile_path = find_benchmark_profile(
            profile_name, config, registry_mgr,
        )
    except ProfileAmbiguousError as e:
        click.echo("Error: %s" % e, err=True)
        click.echo("", err=True)
        click.echo("Matching registries:", err=True)
        for reg_name, path in e.matches:
            click.echo("  @%s/%s" % (reg_name, e.name), err=True)
        sys.exit(1)
    except ProfileError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Read and display profile
    try:
        with open(profile_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        click.echo("Error reading profile: %s" % e, err=True)
        sys.exit(1)

    profile_name_display = data.get("name", profile_path.stem)
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("name") and not data.get("name"):
        profile_name_display = metadata["name"]

    description = data.get("description", "")
    if isinstance(metadata, dict) and not description:
        description = metadata.get("description", "")

    click.echo("Profile:     %s" % profile_name_display)
    click.echo("File:        %s" % profile_path.name)
    click.echo("Path:        %s" % profile_path)
    if description:
        click.echo("Description: %s" % description.strip())

    # Show parameters
    known_keys = {"name", "description", "metadata", "framework", "version"}
    framework = data.get("framework", "")
    if framework:
        click.echo("Framework:   %s" % framework)

    params = data.get("parameters", {})
    if params:
        click.echo("")
        click.echo("Parameters:")
        for key, value in params.items():
            click.echo("  %-25s %s" % (key + ":", value))

    # Show any other top-level keys as parameters (for non-metadata profiles)
    extra = {k: v for k, v in data.items() if k not in known_keys and k != "parameters" and k != "metrics"}
    if extra:
        if not params:
            click.echo("")
            click.echo("Parameters:")
        for key, value in extra.items():
            click.echo("  %-25s %s" % (key + ":", value))

    metrics = data.get("metrics", [])
    if metrics:
        click.echo("")
        click.echo("Metrics:")
        for m in metrics:
            click.echo("  - %s" % m)
