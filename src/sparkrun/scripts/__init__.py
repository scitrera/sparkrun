"""Embedded bash scripts for remote execution.

Scripts are stored as .sh files alongside this module and loaded
via :func:`read_script` at runtime.
"""

from __future__ import annotations

from importlib import resources


def read_script(name: str) -> str:
    """Read a bash script from the scripts package.

    Args:
        name: Script filename (e.g. ``"ip_detect.sh"``).

    Returns:
        Script content as a string.
    """
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
