#!/bin/bash
set -uo pipefail

# Distribute a Docker image from this host to target hosts via docker save/load.
# Placeholders filled by Python: {image}, {targets}, {ssh_opts}, {ssh_user}

IMAGE="{image}"
TARGETS="{targets}"
SSH_OPTS="{ssh_opts}"
SSH_USER="{ssh_user}"

echo "Distributing image $IMAGE to targets: $TARGETS"

FAILED=0
for TARGET in $TARGETS; do
    if [ -n "$SSH_USER" ]; then
        DEST="$SSH_USER@$TARGET"
    else
        DEST="$TARGET"
    fi
    echo "  Sending $IMAGE -> $TARGET ..."
    if docker save "$IMAGE" | ssh $SSH_OPTS "$DEST" 'docker load'; then
        echo "  OK: $TARGET"
    else
        echo "  FAILED: $TARGET" >&2
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED target(s) failed" >&2
    exit 1
fi
echo "Image distributed successfully to all targets"
