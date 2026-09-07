#!/bin/bash
# sparkrun RDMA perftest runner.
#
# Runs one or more commands CONCURRENTLY and reports each one's raw output and
# exit status, framed so the interleaved streams can be separated again.
#
# Concurrency is the feature, not an optimisation: a DGX Spark QSFP112 cable
# presents as two RDMA devices, and driving them one at a time measures ~100
# Gb/s on a cable that carries ~196 Gb/s.  Only a simultaneous run sees the
# real aggregate.
#
# Injected by sparkrun.scripts.inject_shell_vars:
#   RDMA_CMDS      newline-separated commands to run concurrently
#   RDMA_TIMEOUT   per-command wall-clock cap, seconds (default 180)
#   RDMA_RETRIES   retries per command (default 0)
#   RDMA_PRE_SLEEP seconds to wait before starting (default 0)
#
# RDMA_RETRIES exists for the CLIENT side of a perftest pair.  The server is
# dispatched over its own SSH connection at the same moment, so the client can
# arrive first and be refused; perftest has no "wait for peer" mode, so the
# retry IS the rendezvous.  Every command runs under `timeout`, so a server
# whose client never shows up exits on its own rather than lingering on the
# host forever.
#
# Not passed through str.format(), so literal braces are safe here.
set -uo pipefail

: "${RDMA_TIMEOUT:=180}"
: "${RDMA_RETRIES:=0}"
: "${RDMA_PRE_SLEEP:=0}"

if [ "$RDMA_PRE_SLEEP" != "0" ]; then
    sleep "$RDMA_PRE_SLEEP"
fi

mapfile -t _CMDS <<< "${RDMA_CMDS:-}"

# --signal=INT so perftest gets a chance to print partial results; --kill-after
# guarantees it dies even if it ignores that.
run_one() {
    local cmd="$1" out="$2" tries=0 rc=0
    while : ; do
        timeout --signal=INT --kill-after=10 "$RDMA_TIMEOUT" bash -c "$cmd" >"$out" 2>&1
        rc=$?
        [ "$rc" -eq 0 ] && return 0
        tries=$((tries + 1))
        if [ "$tries" -gt "$RDMA_RETRIES" ]; then
            return "$rc"
        fi
        sleep 2
    done
}

_pids=()
_outs=()
_count=0

for _cmd in "${_CMDS[@]}"; do
    [ -z "$_cmd" ] && continue
    _out=$(mktemp)
    _outs+=("$_out")
    run_one "$_cmd" "$_out" &
    _pids+=("$!")
    _count=$((_count + 1))
done

_i=0
for _pid in "${_pids[@]}"; do
    wait "$_pid"
    _rc=$?
    echo "SPARKRUN_RDMA_RUN_${_i}_BEGIN"
    cat "${_outs[$_i]}" 2>/dev/null
    echo "SPARKRUN_RDMA_RUN_${_i}_END"
    echo "SPARKRUN_RDMA_RUN_${_i}_RC=${_rc}"
    rm -f "${_outs[$_i]}"
    _i=$((_i + 1))
done

echo "RDMA_RUN_COUNT=${_count}"
echo "RDMA_COMPLETE=1"
