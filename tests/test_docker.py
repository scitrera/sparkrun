"""Unit tests for sparkrun.orchestration.docker module."""

import pytest
from sparkrun.orchestration.docker import (
    docker_run_cmd,
    docker_exec_cmd,
    docker_stop_cmd,
    docker_inspect_exists_cmd,
    docker_pull_cmd,
    docker_logs_cmd,
    generate_container_name,
)


def test_docker_run_basic():
    """Generate basic docker run with image, verify default opts."""
    cmd = docker_run_cmd("nvcr.io/nvidia/vllm:latest")

    assert cmd.startswith("docker run")
    assert "-d" in cmd  # detach by default
    assert "--privileged" in cmd
    assert "--gpus all" in cmd
    assert "--rm" in cmd
    assert "--ipc=host" in cmd
    assert "--shm-size=10.24gb" in cmd
    assert "--network host" in cmd
    assert "nvcr.io/nvidia/vllm:latest" in cmd


def test_docker_run_detach():
    """Verify -d flag present when detach=True."""
    cmd = docker_run_cmd("test-image:latest", detach=True)
    assert "-d" in cmd


def test_docker_run_no_detach():
    """Verify no -d when detach=False."""
    cmd = docker_run_cmd("test-image:latest", detach=False)
    assert "-d" not in cmd


def test_docker_run_with_name():
    """Verify --name appears."""
    cmd = docker_run_cmd("test-image:latest", container_name="my-container")
    assert "--name my-container" in cmd


def test_docker_run_with_env():
    """Verify -e KEY=VALUE pairs (sorted)."""
    env = {"ZZZ": "last", "AAA": "first", "MMM": "middle"}
    cmd = docker_run_cmd("test-image:latest", env=env)

    # Check all env vars are present
    assert "-e AAA=first" in cmd
    assert "-e MMM=middle" in cmd
    assert "-e ZZZ=last" in cmd

    # Verify sorted order (AAA before MMM before ZZZ)
    aaa_idx = cmd.index("AAA=first")
    mmm_idx = cmd.index("MMM=middle")
    zzz_idx = cmd.index("ZZZ=last")
    assert aaa_idx < mmm_idx < zzz_idx


def test_docker_run_with_volumes():
    """Verify -v host:container pairs (sorted)."""
    volumes = {
        "/host/path3": "/container3",
        "/host/path1": "/container1",
        "/host/path2": "/container2",
    }
    cmd = docker_run_cmd("test-image:latest", volumes=volumes)

    # Check all volumes are present
    assert "-v /host/path1:/container1" in cmd
    assert "-v /host/path2:/container2" in cmd
    assert "-v /host/path3:/container3" in cmd

    # Verify sorted order
    p1_idx = cmd.index("/host/path1:/container1")
    p2_idx = cmd.index("/host/path2:/container2")
    p3_idx = cmd.index("/host/path3:/container3")
    assert p1_idx < p2_idx < p3_idx


def test_docker_run_with_command():
    """Verify command appended at end."""
    cmd = docker_run_cmd("test-image:latest", command="python app.py")
    assert cmd.endswith("python app.py")


def test_docker_run_with_extra_opts():
    """Verify extra options included."""
    extra = ["--ulimit", "nofile=65536", "--cap-add", "SYS_ADMIN"]
    cmd = docker_run_cmd("test-image:latest", extra_opts=extra)

    assert "--ulimit" in cmd
    assert "nofile=65536" in cmd
    assert "--cap-add" in cmd
    assert "SYS_ADMIN" in cmd


def test_docker_exec_basic():
    """Generate basic docker exec."""
    cmd = docker_exec_cmd("my-container", "echo hello")

    assert cmd.startswith("docker exec")
    assert "my-container" in cmd
    assert "bash -c" in cmd
    assert "'echo hello'" in cmd


def test_docker_exec_detach():
    """With detach flag."""
    cmd = docker_exec_cmd("my-container", "echo hello", detach=True)
    assert "-d" in cmd


def test_docker_exec_with_env():
    """With environment variables."""
    env = {"PATH": "/usr/local/bin", "HOME": "/root"}
    cmd = docker_exec_cmd("my-container", "echo hello", env=env)

    # Should be sorted
    assert "-e HOME=/root" in cmd
    assert "-e PATH=/usr/local/bin" in cmd


def test_docker_stop_force():
    """Verify docker rm -f."""
    cmd = docker_stop_cmd("my-container", force=True)
    assert "docker rm -f my-container" in cmd
    assert "2>/dev/null || true" in cmd


def test_docker_stop_graceful():
    """Verify docker stop."""
    cmd = docker_stop_cmd("my-container", force=False)
    assert "docker stop my-container" in cmd
    assert "2>/dev/null || true" in cmd


def test_docker_inspect_exists():
    """Verify inspect command format."""
    cmd = docker_inspect_exists_cmd("test-image:latest")
    assert cmd == "docker image inspect test-image:latest >/dev/null 2>&1"


def test_docker_pull():
    """Verify pull command format."""
    cmd = docker_pull_cmd("nvcr.io/nvidia/vllm:v0.5.3")
    assert cmd == "docker pull nvcr.io/nvidia/vllm:v0.5.3"


def test_docker_logs_basic():
    """Basic logs command."""
    cmd = docker_logs_cmd("my-container")
    assert cmd == "docker logs my-container"


def test_docker_logs_follow_tail():
    """With -f and --tail options."""
    cmd = docker_logs_cmd("my-container", follow=True, tail=100)
    assert cmd == "docker logs -f --tail 100 my-container"


def test_generate_container_name():
    """Test name generation for head, worker, solo roles."""
    assert generate_container_name("sparkrun0", "head") == "sparkrun0_head"
    assert generate_container_name("sparkrun0", "worker") == "sparkrun0_worker"
    assert generate_container_name("cluster-abc", "solo") == "cluster-abc_solo"
