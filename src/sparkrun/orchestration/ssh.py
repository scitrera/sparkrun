"""SSH remote execution via bash -s stdin piping.

All remote operations in sparkrun are executed by generating scripts
as Python strings and piping them to `ssh <host> bash -s` via stdin.
No files are ever copied to remote hosts.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RemoteResult:
    """Result of a remote script execution."""

    host: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def last_line(self) -> str:
        """Get the last non-empty line of stdout (useful for extracting IPs etc)."""
        lines = [line for line in self.stdout.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""


def build_ssh_cmd(
    host: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
) -> list[str]:
    """Build the base SSH command with standard options.

    Args:
        host: Remote hostname or IP address.
        ssh_user: Optional SSH username (prepended as user@host).
        ssh_key: Optional path to SSH private key file.
        ssh_options: Additional SSH command-line options.
        connect_timeout: SSH connection timeout in seconds.

    Returns:
        List of command parts suitable for subprocess.
    """
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    if ssh_options:
        cmd.extend(ssh_options)
    target = f"{ssh_user}@{host}" if ssh_user else host
    cmd.append(target)
    return cmd


def run_remote_script(
    host: str,
    script: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
) -> RemoteResult:
    """Execute a script on a remote host via stdin piping.

    The script is generated in-process and piped directly to
    ``ssh <host> bash -s`` on the remote. No files are copied.

    Args:
        host: Remote hostname or IP.
        script: Bash script content to execute.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the script but don't execute.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    if dry_run:
        logger.info("[dry-run] Would execute on %s:\n%s", host, script)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    cmd = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options, connect_timeout)
    cmd.extend(["bash", "-s"])

    logger.info("  SSH script -> %s (%d bytes)%s",
                host, len(script),
                f" [timeout={timeout}s]" if timeout else "")
    logger.debug("SSH command: %s", " ".join(cmd))
    logger.debug("Script content:\n%s", script)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if result.success:
            logger.info("  SSH script <- %s OK (%.1fs)", host, elapsed)
            if proc.stdout.strip():
                logger.debug("Remote script stdout on %s:\n%s", host, proc.stdout.strip())
            if proc.stderr.strip():
                logger.debug("Remote script stderr on %s:\n%s", host, proc.stderr.strip())
        else:
            logger.warning(
                "  SSH script <- %s FAILED rc=%d (%.1fs): %s",
                host,
                proc.returncode,
                elapsed,
                proc.stderr.strip()[:200],
            )
            if proc.stdout.strip():
                logger.debug("Remote script stdout on %s:\n%s", host, proc.stdout.strip())
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  SSH script <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  SSH script <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


def run_remote_command(
    host: str,
    command: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    connect_timeout: int = 10,
    timeout: int | None = None,
    dry_run: bool = False,
) -> RemoteResult:
    """Execute a single command on a remote host (not via bash -s).

    For simple one-liners where piping a script is overkill.

    Args:
        host: Remote hostname or IP.
        command: Command string to execute remotely.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        connect_timeout: SSH connection timeout in seconds.
        timeout: Overall execution timeout in seconds.
        dry_run: If True, log the command but don't execute.

    Returns:
        RemoteResult with returncode, stdout, stderr.
    """
    if dry_run:
        logger.info("[dry-run] Would run on %s: %s", host, command)
        return RemoteResult(host=host, returncode=0, stdout="[dry-run]", stderr="")

    cmd = build_ssh_cmd(host, ssh_user, ssh_key, ssh_options, connect_timeout)
    cmd.append(command)

    logger.info("  SSH cmd -> %s: %s", host, command[:80])
    logger.debug("SSH command: %s", " ".join(cmd))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - t0
        result = RemoteResult(
            host=host,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        logger.info("  SSH cmd <- %s rc=%d (%.1fs)", host, proc.returncode, elapsed)
        if proc.stdout.strip():
            logger.debug("Remote command stdout on %s:\n%s", host, proc.stdout.strip())
        if proc.stderr.strip():
            logger.debug("Remote command stderr on %s:\n%s", host, proc.stderr.strip())
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.error("  SSH cmd <- %s TIMEOUT after %.0fs", host, elapsed)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr="Execution timed out")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("  SSH cmd <- %s ERROR (%.1fs): %s", host, elapsed, e)
        return RemoteResult(host=host, returncode=-1, stdout="", stderr=str(e))


def run_remote_scripts_parallel(
    hosts: list[str],
    script: str,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
    timeout: int | None = None,
    dry_run: bool = False,
) -> list[RemoteResult]:
    """Execute the same script on multiple hosts in parallel using threads.

    Args:
        hosts: List of remote hostnames or IPs.
        script: Bash script content to execute on each host.
        ssh_user: Optional SSH username.
        ssh_key: Optional path to SSH private key.
        ssh_options: Additional SSH options.
        timeout: Per-host execution timeout in seconds.
        dry_run: If True, log the script but don't execute.

    Returns:
        List of RemoteResult, one per host (order not guaranteed).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Running script in parallel on %d hosts: %s",
                len(hosts), ", ".join(hosts))

    t0 = time.monotonic()
    results: list[RemoteResult] = []
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {
            executor.submit(
                run_remote_script,
                host,
                script,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
                ssh_options=ssh_options,
                timeout=timeout,
                dry_run=dry_run,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if r.success)
    logger.info("  Parallel execution done: %d/%d OK (%.1fs total)",
                ok, len(results), elapsed)

    return results
