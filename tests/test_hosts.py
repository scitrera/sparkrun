"""Tests for sparkrun.hosts module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkrun.hosts import resolve_hosts, parse_hosts_file, HostResolutionError
from sparkrun.cluster_manager import ClusterManager


def test_parse_hosts_file_basic(tmp_path: Path):
    """Parse simple file with one host per line."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text("host1\nhost2\nhost3\n")

    result = parse_hosts_file(hosts_file)
    assert result == ["host1", "host2", "host3"]


def test_parse_hosts_file_with_comments(tmp_path: Path):
    """Lines starting with # are skipped."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text(
        "# This is a comment\n"
        "host1\n"
        "# Another comment\n"
        "host2\n"
    )

    result = parse_hosts_file(hosts_file)
    assert result == ["host1", "host2"]


def test_parse_hosts_file_inline_comments(tmp_path: Path):
    """host # comment strips comment portion."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text(
        "host1  # production server\n"
        "host2# dev server\n"
        "host3\n"
    )

    result = parse_hosts_file(hosts_file)
    assert result == ["host1", "host2", "host3"]


def test_parse_hosts_file_blank_lines(tmp_path: Path):
    """Blank lines are skipped."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text(
        "\n"
        "host1\n"
        "\n"
        "\n"
        "host2\n"
        "   \n"
        "host3\n"
        "\n"
    )

    result = parse_hosts_file(hosts_file)
    assert result == ["host1", "host2", "host3"]


def test_parse_hosts_file_not_found(tmp_path: Path):
    """Raises HostResolutionError when file not found."""
    nonexistent = tmp_path / "nonexistent.txt"

    with pytest.raises(HostResolutionError, match="Hosts file not found"):
        parse_hosts_file(nonexistent)


def test_resolve_hosts_from_cli_string():
    """hosts="a,b,c" returns ["a","b","c"]."""
    result = resolve_hosts(hosts="host1,host2,host3")
    assert result == ["host1", "host2", "host3"]


def test_resolve_hosts_from_cli_string_with_whitespace():
    """hosts string with extra whitespace is stripped."""
    result = resolve_hosts(hosts=" host1 , host2  ,  host3 ")
    assert result == ["host1", "host2", "host3"]


def test_resolve_hosts_from_file(tmp_path: Path):
    """hosts_file takes priority when hosts is None."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text("filehost1\nfilehost2\n")

    result = resolve_hosts(hosts_file=str(hosts_file))
    assert result == ["filehost1", "filehost2"]


def test_resolve_hosts_from_cluster_name(tmp_path: Path):
    """cluster_name lookup via ClusterManager."""
    cm = ClusterManager(tmp_path)
    cm.create("testcluster", hosts=["clusterhost1", "clusterhost2"])

    result = resolve_hosts(cluster_name="testcluster", cluster_manager=cm)
    assert result == ["clusterhost1", "clusterhost2"]


def test_resolve_hosts_from_default_cluster(tmp_path: Path):
    """Falls back to default cluster."""
    cm = ClusterManager(tmp_path)
    cm.create("default_cluster", hosts=["defaulthost1", "defaulthost2"])
    cm.set_default("default_cluster")

    result = resolve_hosts(cluster_manager=cm)
    assert result == ["defaulthost1", "defaulthost2"]


def test_resolve_hosts_from_config_defaults():
    """Falls back to config_default_hosts."""
    result = resolve_hosts(config_default_hosts=["confighost1", "confighost2"])
    assert result == ["confighost1", "confighost2"]


def test_resolve_hosts_empty_when_nothing():
    """Returns empty list when no source."""
    result = resolve_hosts()
    assert result == []


def test_resolve_hosts_cli_overrides_file(tmp_path: Path):
    """hosts string takes priority over file."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text("filehost1\nfilehost2\n")

    result = resolve_hosts(hosts="clihost1,clihost2", hosts_file=str(hosts_file))
    assert result == ["clihost1", "clihost2"]


def test_resolve_hosts_file_overrides_cluster(tmp_path: Path):
    """file takes priority over cluster name."""
    cm = ClusterManager(tmp_path)
    cm.create("testcluster", hosts=["clusterhost1", "clusterhost2"])

    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text("filehost1\nfilehost2\n")

    result = resolve_hosts(
        hosts_file=str(hosts_file),
        cluster_name="testcluster",
        cluster_manager=cm,
    )
    assert result == ["filehost1", "filehost2"]


def test_resolve_hosts_cluster_overrides_config_defaults(tmp_path: Path):
    """cluster_name takes priority over config defaults."""
    cm = ClusterManager(tmp_path)
    cm.create("testcluster", hosts=["clusterhost1", "clusterhost2"])

    result = resolve_hosts(
        cluster_name="testcluster",
        cluster_manager=cm,
        config_default_hosts=["confighost1", "confighost2"],
    )
    assert result == ["clusterhost1", "clusterhost2"]


def test_resolve_hosts_default_cluster_overrides_config_defaults(tmp_path: Path):
    """Default cluster takes priority over config defaults."""
    cm = ClusterManager(tmp_path)
    cm.create("default_cluster", hosts=["defaulthost1", "defaulthost2"])
    cm.set_default("default_cluster")

    result = resolve_hosts(
        cluster_manager=cm,
        config_default_hosts=["confighost1", "confighost2"],
    )
    assert result == ["defaulthost1", "defaulthost2"]


def test_resolve_hosts_cluster_not_found_falls_back(tmp_path: Path):
    """When cluster_name not found, falls back to next priority."""
    cm = ClusterManager(tmp_path)

    result = resolve_hosts(
        cluster_name="nonexistent",
        cluster_manager=cm,
        config_default_hosts=["confighost1"],
    )
    assert result == ["confighost1"]


def test_resolve_hosts_full_priority_chain(tmp_path: Path):
    """Test complete priority chain with all options."""
    cm = ClusterManager(tmp_path)
    cm.create("testcluster", hosts=["clusterhost1"])
    cm.set_default("testcluster")

    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text("filehost1\n")

    # CLI string wins over all
    result = resolve_hosts(
        hosts="clihost1,clihost2",
        hosts_file=str(hosts_file),
        cluster_name="testcluster",
        cluster_manager=cm,
        config_default_hosts=["confighost1"],
    )
    assert result == ["clihost1", "clihost2"]


def test_parse_hosts_file_mixed_content(tmp_path: Path):
    """Test file with mix of valid hosts, comments, and blank lines."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text(
        "# Header comment\n"
        "\n"
        "prod-node-01  # Primary\n"
        "prod-node-02\n"
        "\n"
        "# Section: backup nodes\n"
        "backup-node-01\n"
        "   \n"
        "backup-node-02 # Secondary backup\n"
        "\n"
    )

    result = parse_hosts_file(hosts_file)
    assert result == [
        "prod-node-01",
        "prod-node-02",
        "backup-node-01",
        "backup-node-02",
    ]


def test_resolve_hosts_empty_string_skips_to_next_priority(tmp_path: Path):
    """Empty hosts string should skip to next priority."""
    hosts_file = tmp_path / "hosts.txt"
    hosts_file.write_text("filehost1\n")

    # Empty string should not be treated as valid input
    result = resolve_hosts(hosts="", hosts_file=str(hosts_file))
    assert result == ["filehost1"]


def test_resolve_hosts_whitespace_only_string_returns_empty():
    """hosts string with only whitespace/commas returns empty list."""
    result = resolve_hosts(hosts="  ,  ,  ")
    assert result == []
