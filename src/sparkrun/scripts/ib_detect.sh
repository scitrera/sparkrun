#!/bin/bash
# InfiniBand/RDMA detection for DGX Spark
# Outputs key=value pairs for NCCL configuration

set -uo pipefail

# Check for RDMA devices
if ! command -v ibstat &>/dev/null && ! [ -d /sys/class/infiniband ]; then
    echo "IB_DETECTED=0"
    exit 0
fi

# Find active InfiniBand ports
IB_DEVICES=()
if [ -d /sys/class/infiniband ]; then
    for dev in /sys/class/infiniband/*/ports/*/state; do
        if [ -f "$dev" ] && grep -q "ACTIVE" "$dev" 2>/dev/null; then
            dev_path=$(dirname $(dirname "$dev"))
            dev_name=$(basename "$dev_path")
            port_num=$(basename $(dirname "$dev"))
            IB_DEVICES+=("${dev_name}:${port_num}")
        fi
    done
fi

if [ ${#IB_DEVICES[@]} -eq 0 ]; then
    echo "IB_DETECTED=0"
    exit 0
fi

# Build HCA list
HCA_LIST=""
for dev_port in "${IB_DEVICES[@]}"; do
    dev_name="${dev_port%%:*}"
    if [ -z "$HCA_LIST" ]; then
        HCA_LIST="$dev_name"
    else
        HCA_LIST="$HCA_LIST,$dev_name"
    fi
done

# Detect GID index (prefer RoCE v2 = GID index 3, fall back to 0)
FIRST_DEV="${IB_DEVICES[0]}"
DEV_NAME="${FIRST_DEV%%:*}"
PORT_NUM="${FIRST_DEV##*:}"
GID_INDEX=0
GID_DIR="/sys/class/infiniband/${DEV_NAME}/ports/${PORT_NUM}/gids"
if [ -d "$GID_DIR" ]; then
    for gid_file in "$GID_DIR"/*; do
        idx=$(basename "$gid_file")
        gid_val=$(cat "$gid_file" 2>/dev/null || true)
        type_file="/sys/class/infiniband/${DEV_NAME}/ports/${PORT_NUM}/gid_attrs/types/${idx}"
        if [ -f "$type_file" ]; then
            gid_type=$(cat "$type_file" 2>/dev/null || true)
            if [[ "$gid_type" == *"RoCE v2"* ]] || [[ "$gid_type" == *"RoCEv2"* ]]; then
                GID_INDEX="$idx"
                break
            fi
        fi
    done
fi

# Detect management network interface (default route interface)
DEFAULT_IF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")

# Build net device list from RDMA links
NET_LIST=""
UCX_LIST=""
if command -v rdma &>/dev/null; then
    while IFS= read -r line; do
        netdev=$(echo "$line" | grep -oP 'netdev \K\S+' || true)
        rdma_dev=$(echo "$line" | sed -n 's|^\([0-9]*\): \([^ ]*\)/.*|\2|p' || true)
        if [ -n "$netdev" ]; then
            if [ -z "$NET_LIST" ]; then
                NET_LIST="$netdev"
            else
                NET_LIST="$NET_LIST,$netdev"
            fi
        fi
        if [ -n "$rdma_dev" ]; then
            port=$(echo "$line" | sed -n 's|.*/\([0-9]*\).*|\1|p' || true)
            ucx_entry="${rdma_dev}:${port:-1}"
            if [ -z "$UCX_LIST" ]; then
                UCX_LIST="$ucx_entry"
            else
                UCX_LIST="$UCX_LIST,$ucx_entry"
            fi
        fi
    done < <(rdma link show 2>/dev/null || true)
fi

# Fall back to default interface if no RDMA net devices found
if [ -z "$NET_LIST" ]; then
    NET_LIST="$DEFAULT_IF"
fi

echo "IB_DETECTED=1"
echo "DETECTED_GID_INDEX=$GID_INDEX"
echo "DETECTED_HCA_LIST=$HCA_LIST"
echo "DETECTED_SOCKET_IFNAME=$DEFAULT_IF"
echo "DETECTED_NET_LIST=$NET_LIST"
echo "DETECTED_UCX_LIST=$UCX_LIST"
