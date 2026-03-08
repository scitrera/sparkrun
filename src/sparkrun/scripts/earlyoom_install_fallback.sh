#!/bin/bash
# Install and configure earlyoom with sparkrun-optimized settings.
# Runs as root via run_remote_sudo_script (no sudo prefix needed).
# Params: {prefer} {avoid}
set -euo pipefail

PREFER="{prefer}"
AVOID="{avoid}"

# Install earlyoom if not present
if ! command -v earlyoom >/dev/null 2>&1; then
    echo "INSTALLING: earlyoom (apt-get update + install, may take ~1 minute)"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq earlyoom
    echo "INSTALLED: earlyoom"
else
    echo "PRESENT: earlyoom already installed"
fi

# Configure /etc/default/earlyoom
EARLYOOM_CONF="/etc/default/earlyoom"
cat > "$EARLYOOM_CONF" << CONF_EOF
# Configured by: sparkrun setup earlyoom
# -m 3    : trigger at 3% available memory
# -s 10   : trigger at 10% available swap
# --prefer: processes to kill first on OOM (inference workloads)
# --avoid : processes to protect from OOM kill
EARLYOOM_ARGS="-m 3 -s 10 --prefer '$PREFER' --avoid '$AVOID'"
CONF_EOF
echo "CONFIGURED: $EARLYOOM_CONF"

# Install systemd override for CAP_KILL + SIGKILL behavior
OVERRIDE_DIR="/etc/systemd/system/earlyoom.service.d"
mkdir -p "$OVERRIDE_DIR"
cat > "$OVERRIDE_DIR/override.conf" << OVERRIDE_EOF
[Service]
CapabilityBoundingSet=CAP_KILL CAP_IPC_LOCK
AmbientCapabilities=CAP_KILL CAP_IPC_LOCK
ExecStart=
ExecStart=/usr/bin/earlyoom \$EARLYOOM_ARGS
OVERRIDE_EOF
echo "CONFIGURED: systemd override"

# Reload and restart
systemctl daemon-reload
systemctl enable earlyoom
systemctl restart earlyoom

# Verify
if systemctl is-active --quiet earlyoom; then
    echo "OK: earlyoom running"
else
    echo "ERROR: earlyoom failed to start"
    exit 1
fi
