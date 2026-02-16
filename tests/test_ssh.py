"""Unit tests for sparkrun.orchestration.ssh module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from sparkrun.orchestration.ssh import (
    build_ssh_cmd,
    RemoteResult,
    run_remote_script,
    run_remote_command,
    run_remote_scripts_parallel,
)


def test_build_ssh_cmd_basic():
    """Basic SSH command with just host."""
    cmd = build_ssh_cmd("192.168.1.100")

    assert cmd[0] == "ssh"
    assert "-o" in cmd
    assert "BatchMode=yes" in cmd
    assert "ConnectTimeout=10" in cmd
    assert "192.168.1.100" in cmd


def test_build_ssh_cmd_with_user():
    """With ssh_user."""
    cmd = build_ssh_cmd("192.168.1.100", ssh_user="root")

    assert "root@192.168.1.100" in cmd


def test_build_ssh_cmd_with_key():
    """With ssh_key."""
    cmd = build_ssh_cmd("192.168.1.100", ssh_key="/path/to/key.pem")

    assert "-i" in cmd
    assert "/path/to/key.pem" in cmd


def test_build_ssh_cmd_with_options():
    """With extra ssh_options."""
    cmd = build_ssh_cmd("192.168.1.100", ssh_options=["-v", "-o", "StrictHostKeyChecking=no"])

    assert "-v" in cmd
    assert "StrictHostKeyChecking=no" in cmd


def test_remote_result_success():
    """RemoteResult with returncode=0 is success."""
    result = RemoteResult(host="host1", returncode=0, stdout="OK", stderr="")
    assert result.success is True


def test_remote_result_failure():
    """RemoteResult with returncode=1 is not success."""
    result = RemoteResult(host="host1", returncode=1, stdout="", stderr="error")
    assert result.success is False


def test_remote_result_last_line():
    """Test last_line extraction from stdout."""
    result = RemoteResult(
        host="host1",
        returncode=0,
        stdout="line1\nline2\nline3\n",
        stderr="",
    )
    assert result.last_line == "line3"


def test_remote_result_last_line_empty():
    """Empty stdout returns empty string."""
    result = RemoteResult(host="host1", returncode=0, stdout="", stderr="")
    assert result.last_line == ""


def test_remote_result_last_line_with_blank_lines():
    """Test last_line ignores trailing blank lines."""
    result = RemoteResult(
        host="host1",
        returncode=0,
        stdout="line1\nline2\n\n\n",
        stderr="",
    )
    assert result.last_line == "line2"


def test_run_remote_script_dry_run():
    """Dry run returns success without subprocess."""
    result = run_remote_script(
        "192.168.1.100",
        "#!/bin/bash\necho test",
        dry_run=True,
    )

    assert result.success
    assert result.host == "192.168.1.100"
    assert result.stdout == "[dry-run]"
    assert result.returncode == 0


def test_run_remote_command_dry_run():
    """Dry run returns success without subprocess."""
    result = run_remote_command(
        "192.168.1.100",
        "echo test",
        dry_run=True,
    )

    assert result.success
    assert result.host == "192.168.1.100"
    assert result.stdout == "[dry-run]"
    assert result.returncode == 0


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_script_mocks_subprocess(mock_run):
    """Mock subprocess.run, verify ssh bash -s is called with script as input."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "output"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    script = "#!/bin/bash\necho hello"
    result = run_remote_script("192.168.1.100", script)

    # Verify subprocess.run was called
    assert mock_run.called
    call_args = mock_run.call_args

    # Check command structure
    cmd = call_args[0][0]
    assert cmd[0] == "ssh"
    assert "bash" in cmd
    assert "-s" in cmd

    # Check script was passed as input
    assert call_args[1]["input"] == script
    assert call_args[1]["text"] is True
    assert call_args[1]["capture_output"] is True

    # Verify result
    assert result.success
    assert result.stdout == "output"


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_command_mocks_subprocess(mock_run):
    """Mock subprocess.run, verify command execution."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "hello"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    result = run_remote_command("192.168.1.100", "echo hello")

    # Verify subprocess.run was called
    assert mock_run.called
    call_args = mock_run.call_args

    # Check command structure
    cmd = call_args[0][0]
    assert cmd[0] == "ssh"
    assert "echo hello" in cmd

    # Verify result
    assert result.success
    assert result.stdout == "hello"


def test_run_remote_scripts_parallel_dry_run():
    """Parallel dry run for multiple hosts."""
    hosts = ["host1", "host2", "host3"]
    script = "#!/bin/bash\necho test"

    results = run_remote_scripts_parallel(hosts, script, dry_run=True)

    assert len(results) == 3
    assert all(r.success for r in results)
    assert {r.host for r in results} == set(hosts)


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_scripts_parallel_mocks_subprocess(mock_run):
    """Test parallel execution with mocked subprocess."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "output"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    hosts = ["host1", "host2"]
    script = "#!/bin/bash\necho test"

    results = run_remote_scripts_parallel(hosts, script)

    # Should have called subprocess.run twice (once per host)
    assert mock_run.call_count == 2
    assert len(results) == 2
    assert all(r.success for r in results)


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_script_timeout(mock_run):
    """Test timeout handling."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=10)

    result = run_remote_script("192.168.1.100", "sleep 100", timeout=10)

    assert not result.success
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


@patch("sparkrun.orchestration.ssh.subprocess.run")
def test_run_remote_script_exception(mock_run):
    """Test exception handling."""
    mock_run.side_effect = Exception("Connection refused")

    result = run_remote_script("192.168.1.100", "echo test")

    assert not result.success
    assert result.returncode == -1
    assert "Connection refused" in result.stderr
