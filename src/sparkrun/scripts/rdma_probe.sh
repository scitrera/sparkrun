#!/bin/bash
# sparkrun RDMA readiness probe.
#
# Emits key=value pairs on stdout; NEVER modifies host state.  Diagnostic
# noise goes to stderr so it cannot corrupt the key=value stream.
#
# Answers three questions `sparkrun setup rdma-test` has to settle before it
# can run anything: are the RDMA devices actually up, is perftest already on
# the host (DGX OS ships it, in which case the bandwidth/latency suite needs
# no container at all), and is docker available for the NCCL suite.
#
# The per-device `rate` is load-bearing beyond diagnostics: it is where the
# bandwidth expectation comes from, so the verdict adapts to the hardware
# instead of hardcoding a DGX Spark number that would be wrong everywhere
# else.
#
# Takes no parameters and is not passed through str.format(), so literal
# braces are safe here.
set -uo pipefail

SYSFS_IB="${SPARKRUN_IB_SYSFS:-/sys/class/infiniband}"

if command -v ib_write_bw >/dev/null 2>&1; then
    echo "RDMA_PERFTEST=1"
else
    echo "RDMA_PERFTEST=0"
fi

if command -v docker >/dev/null 2>&1; then
    echo "RDMA_DOCKER=1"
else
    echo "RDMA_DOCKER=0"
fi

# Best-effort RoCEv2 IPv4 GID index.  Deliberately NOT a reimplementation of
# ib_detect.sh's resolution chain: with rdma_cm (-R) perftest picks the GID
# itself, so this is only consulted when an operator overrides it.
GID_INDEX=""
if command -v show_gids >/dev/null 2>&1; then
    GID_INDEX=$(show_gids 2>/dev/null | awk '$5 ~ /v2/ && $4 ~ /\./ {print $3; exit}')
fi
echo "RDMA_GID_INDEX=$GID_INDEX"

DEV_COUNT=0
if [ -d "$SYSFS_IB" ]; then
    for dev_path in "$SYSFS_IB"/*; do
        [ -d "$dev_path" ] || continue
        dev_name=$(basename "$dev_path")

        state=""
        [ -r "$dev_path/ports/1/state" ] && state=$(cat "$dev_path/ports/1/state" 2>/dev/null)

        rate=""
        [ -r "$dev_path/ports/1/rate" ] && rate=$(cat "$dev_path/ports/1/rate" 2>/dev/null)

        link_layer=""
        [ -r "$dev_path/ports/1/link_layer" ] && link_layer=$(cat "$dev_path/ports/1/link_layer" 2>/dev/null)

        netdev=""
        if [ -d "$dev_path/device/net" ]; then
            for net_path in "$dev_path/device/net"/*; do
                [ -e "$net_path" ] || continue
                netdev=$(basename "$net_path")
                break
            done
        fi

        # Physical-port identity. Several RDMA devices can be functions of one
        # adapter, and on DGX Spark several are functions of one adapter
        # sharing a single physical port — so their bandwidths are NOT
        # additive. sys_image_guid names the adapter; phys_port_name (from the
        # netdev) names the port on it. Together they identify the wire.
        sys_image_guid=""
        [ -r "$dev_path/sys_image_guid" ] && sys_image_guid=$(cat "$dev_path/sys_image_guid" 2>/dev/null)

        phys_port_name=""
        if [ -n "$netdev" ] && [ -r "/sys/class/net/$netdev/phys_port_name" ]; then
            phys_port_name=$(cat "/sys/class/net/$netdev/phys_port_name" 2>/dev/null)
        fi

        echo "RDMA_DEV_${DEV_COUNT}_NAME=$dev_name"
        echo "RDMA_DEV_${DEV_COUNT}_STATE=$state"
        echo "RDMA_DEV_${DEV_COUNT}_RATE=$rate"
        echo "RDMA_DEV_${DEV_COUNT}_LINK_LAYER=$link_layer"
        echo "RDMA_DEV_${DEV_COUNT}_NETDEV=$netdev"
        echo "RDMA_DEV_${DEV_COUNT}_SYS_IMAGE_GUID=$sys_image_guid"
        echo "RDMA_DEV_${DEV_COUNT}_PHYS_PORT_NAME=$phys_port_name"
        DEV_COUNT=$((DEV_COUNT + 1))
    done
else
    echo "No $SYSFS_IB directory; host has no RDMA devices" >&2
fi

echo "RDMA_DEV_COUNT=$DEV_COUNT"
echo "RDMA_COMPLETE=1"
