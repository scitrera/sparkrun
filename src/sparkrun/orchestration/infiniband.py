"""InfiniBand/RDMA detection script generator.

Generates a bash script that detects InfiniBand interfaces and outputs
NCCL/RDMA environment variables. The script is piped to remote hosts
via SSH bash -s.
"""

from __future__ import annotations

import logging

from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)


def generate_ib_detect_script() -> str:
    """Generate a bash script that detects InfiniBand interfaces.

    The script outputs key=value pairs on stdout that can be parsed
    to configure NCCL and RDMA settings for multi-node inference.

    Output variables (if IB is found)::

        DETECTED_GID_INDEX=<n>
        DETECTED_HCA_LIST=<comma-separated HCA names>
        DETECTED_SOCKET_IFNAME=<interface>
        DETECTED_NET_LIST=<comma-separated net interfaces>
        DETECTED_UCX_LIST=<comma-separated UCX devices>
        IB_DETECTED=1

    If no IB is found, outputs::

        IB_DETECTED=0

    Returns:
        Bash script content as a string.
    """
    return read_script("ib_detect.sh")


def parse_ib_detect_output(output: str) -> dict[str, str]:
    """Parse the output of the IB detection script into a dict.

    Args:
        output: Raw stdout from the IB detection script.

    Returns:
        Dictionary of detected key=value pairs.
    """
    result: dict[str, str] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def generate_nccl_env(ib_info: dict[str, str]) -> dict[str, str]:
    """Generate NCCL environment variables from IB detection results.

    Args:
        ib_info: Parsed output from :func:`parse_ib_detect_output`.

    Returns:
        Dictionary of NCCL/network environment variables.
        Empty dict if no InfiniBand was detected.
    """
    if ib_info.get("IB_DETECTED") != "1":
        return {}

    env: dict[str, str] = {
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_CROSS_NIC": "1",
    }

    if ib_info.get("DETECTED_GID_INDEX"):
        env["NCCL_IB_GID_INDEX"] = ib_info["DETECTED_GID_INDEX"]
    if ib_info.get("DETECTED_HCA_LIST"):
        env["NCCL_IB_HCA"] = ib_info["DETECTED_HCA_LIST"]
    if ib_info.get("DETECTED_SOCKET_IFNAME"):
        env["NCCL_SOCKET_IFNAME"] = ib_info["DETECTED_SOCKET_IFNAME"]
    if ib_info.get("DETECTED_NET_LIST"):
        net_list = ib_info["DETECTED_NET_LIST"]
        env["MN_IF_NAME"] = net_list
        env["OMPI_MCA_btl_tcp_if_include"] = net_list
        env["GLOO_SOCKET_IFNAME"] = net_list
        env["TP_SOCKET_IFNAME"] = net_list
    if ib_info.get("DETECTED_UCX_LIST"):
        env["UCX_NET_DEVICES"] = ib_info["DETECTED_UCX_LIST"]

    return env
