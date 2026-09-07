"""Public setup errors for :mod:`sparkrun.api.setup`.

Local-tooling and SSH-bootstrap failures are translated into these
:class:`~sparkrun.api._errors.SparkrunError` subclasses at the api boundary, so
callers can ``except SparkrunError`` uniformly.
"""

from __future__ import annotations

from sparkrun.api._errors import SparkrunError


class SshAccessError(SparkrunError):
    """SSH access to the cluster could not be established or verified."""


class SshKeyError(SshAccessError):
    """A local SSH key could not be located or generated."""


class OpenSshUnavailable(SshAccessError):
    """The OpenSSH client binaries (``ssh`` / ``ssh-keygen``) are missing."""


class RdmaTestError(SparkrunError):
    """The RDMA fabric test could not be set up or run.

    Raised only when the test could not *start* — no usable links, an image
    that would not fetch, a container that would not start. A link that runs
    and measures badly is a finding in the report, not an exception.
    """


__all__ = [
    "SshAccessError",
    "SshKeyError",
    "OpenSshUnavailable",
    "RdmaTestError",
]
