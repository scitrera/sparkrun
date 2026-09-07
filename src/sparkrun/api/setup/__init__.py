"""Public library API for cluster setup operations.

Console-free surface for the parts of ``sparkrun setup`` that a GUI needs to
drive itself.  Mirrors the :mod:`sparkrun.api.k8s` / :mod:`sparkrun.api.tailscale`
conventions: dataclass returns, typed
:class:`~sparkrun.api._errors.SparkrunError` subclasses, no writes to
stdout/stderr.

Today this covers the SSH access bootstrap — the step that has to succeed
before any other setup phase can run, and the one that fails on a control
machine that has never talked to the cluster:

- :func:`probe_ssh_access` — non-interactive reachability + failure diagnosis.
- :func:`ensure_local_key` — find or generate the identity to install.
- :func:`install_public_key_interactive` — install it via password auth.
- :func:`mesh_ssh_keys_native` — host↔host key mesh with no local shell.

Every function here works from a bare Windows control machine, using only the
OpenSSH client binaries.
"""

from __future__ import annotations

from ._errors import OpenSshUnavailable, RdmaTestError, SshAccessError, SshKeyError
from ._mesh import (
    MeshResult,
    build_collect_key_script,
    build_install_keys_script,
    mesh_ssh_keys_native,
)
from ._rdma import (
    DEFAULT_RDMA_TEST_IMAGE,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    STATUS_WARN,
    SUITE_ALL,
    SUITE_NCCL,
    SUITE_PERFTEST,
    LinkTestResult,
    NcclTestResult,
    PairTestResult,
    RdmaTestReport,
    rdma_test,
)
from ._ssh_access import (
    SPARKRUN_KEY_NAME,
    LocalSshKey,
    SshProbe,
    build_authorized_key_script,
    ensure_local_key,
    install_public_key_interactive,
    probe_ssh_access,
)

__all__ = [
    # Functions
    "probe_ssh_access",
    "ensure_local_key",
    "install_public_key_interactive",
    "mesh_ssh_keys_native",
    "rdma_test",
    # Script builders (exposed for testing and for callers that pipe them
    # through their own transport)
    "build_authorized_key_script",
    "build_collect_key_script",
    "build_install_keys_script",
    # Data models
    "SshProbe",
    "LocalSshKey",
    "MeshResult",
    "RdmaTestReport",
    "PairTestResult",
    "LinkTestResult",
    "NcclTestResult",
    # Errors
    "SshAccessError",
    "SshKeyError",
    "OpenSshUnavailable",
    "RdmaTestError",
    # Constants
    "SPARKRUN_KEY_NAME",
    "DEFAULT_RDMA_TEST_IMAGE",
    "SUITE_PERFTEST",
    "SUITE_NCCL",
    "SUITE_ALL",
    "STATUS_OK",
    "STATUS_WARN",
    "STATUS_FAIL",
    "STATUS_SKIP",
]
