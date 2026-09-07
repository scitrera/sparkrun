"""MPI helpers shared by everything that runs ``mpirun`` across containers.

Open MPI launches remote ranks by shelling out to an *rsh agent* — by default
``ssh``, which lands the rank on the peer **host**.  Every sparkrun workload
runs inside a container, so the rank has to land one level deeper.

There are two ways to arrange that and only one of them is acceptable here:
install an sshd inside every container and have mpirun connect to it on a
non-standard port, or keep using the host's existing sshd and step into the
container on arrival.  The second needs no daemon, no extra port, no key
material inside the image and no second authentication surface — so
:func:`build_rsh_wrapper` generates a tiny agent that SSHes to the peer host
and ``docker exec``s the command into that host's container.

Shared rather than per-caller because two copies would drift, and the symptom
of drift is a multi-node job that hangs at rank startup with no error — the
most expensive kind of bug to chase.
"""

from __future__ import annotations

import re

__all__ = [
    "DEFAULT_RSH_WRAPPER_PATH",
    "DEFAULT_CONTAINER_SSH_KEY",
    "build_rsh_wrapper",
]

#: Where the generated agent is written inside the head container.
DEFAULT_RSH_WRAPPER_PATH = "/tmp/sparkrun-rsh-wrapper.sh"

#: Private key inside the container, from the mounted ``~/.ssh``.
DEFAULT_CONTAINER_SSH_KEY = "/tmp/.ssh/id_ed25519"

# The address is emitted bare as a `case` pattern and the other two inside
# double quotes, so none of them can be shell-quoted without changing what
# bash sees.  They are validated instead.  Every value here is sparkrun's own
# (a resolved host address, a container name it generated, a path it chose) —
# these guards are a tripwire against a future caller passing something
# attacker-influenced, not a filter for routine input.
_SAFE_ADDRESS = re.compile(r"\A[A-Za-z0-9._:-]+\Z")
_SAFE_CONTAINER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_PATH = re.compile(r"\A[A-Za-z0-9_./-]+\Z")
_SAFE_USER = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.-]*\Z")


def build_rsh_wrapper(
    host_container_map: dict[str, str],
    ssh_key_path: str = DEFAULT_CONTAINER_SSH_KEY,
    ssh_user: str | None = None,
) -> str:
    """Generate the bash rsh agent for ``mpirun --mca plm_rsh_agent``.

    Open MPI invokes the agent as ``<agent> <host> <command...>``.  The
    generated script maps *host* to that host's container name and re-execs
    through the host's own sshd into the container.

    Args:
        host_container_map: Address mpirun will name (as it appears in
            ``-H``) → container name on that host.  The keys must match the
            ``-H`` list exactly; mpirun passes them through verbatim, so a
            hostname there and an IP here will not match.
        ssh_key_path: Private key **inside the container**.
        ssh_user: Username for the outbound SSH.  **Supply this whenever the
            container user differs from the cluster's SSH user**, which is
            the normal case: containers run as ``root`` while the cluster is
            reached as an ordinary login, so an unqualified ``ssh <host>``
            goes out as ``root@`` and is refused. mpirun reports that as
            "lack of authority to execute on one or more specified nodes",
            which points nowhere near the cause. ``None`` preserves the
            historical unqualified form.

    Returns:
        The complete bash script.

    Raises:
        ValueError: An address, container name, key path or username
            contains characters that cannot be safely interpolated.
    """
    if not _SAFE_PATH.match(ssh_key_path):
        raise ValueError("unsafe ssh key path for rsh agent: %r" % ssh_key_path)
    if ssh_user is not None and not _SAFE_USER.match(ssh_user):
        raise ValueError("unsafe ssh user for rsh agent: %r" % ssh_user)
    target = '%s@"$HOST"' % ssh_user if ssh_user else '"$HOST"'

    lines = [
        "#!/bin/bash",
        "# mpirun rsh agent: SSH to worker HOST, docker exec into container",
        "HOST=$1; shift",
        "case $HOST in",
    ]
    for address, container_name in sorted(host_container_map.items()):
        if not _SAFE_ADDRESS.match(address):
            raise ValueError("unsafe host address for rsh agent: %r" % address)
        if not _SAFE_CONTAINER.match(container_name):
            raise ValueError("unsafe container name for rsh agent: %r" % container_name)
        lines.append('    %s) CONTAINER="%s" ;;' % (address, container_name))
    lines.extend(
        [
            '    *) echo "Unknown host: $HOST" >&2; exit 1 ;;',
            "esac",
            # TOFU: accept-new matches the codebase convention and still
            # protects against MITM after the first connection.
            "exec ssh -o StrictHostKeyChecking=accept-new \\",
            '  -i %s %s docker exec "$CONTAINER" "$@"' % (ssh_key_path, target),
        ]
    )
    return "\n".join(lines) + "\n"
