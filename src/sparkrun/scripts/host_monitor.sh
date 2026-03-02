#!/bin/bash
# host_monitor.sh — Lightweight continuous system monitoring for DGX Spark.
#
# Outputs CSV with a header row followed by one data row per sample at
# INTERVAL-second intervals.  Designed to run over SSH with minimal
# resource footprint — uses nvidia-smi, /proc, and /sys exclusively.
#
# Configuration (env vars or positional arg):
#   MONITOR_INTERVAL  Seconds between samples (default: 2)
#   $1                Overrides MONITOR_INTERVAL if provided
#
# CSV columns:
#   timestamp           ISO-8601 UTC timestamp
#   hostname            Machine hostname
#   uptime_sec          System uptime in seconds
#   cpu_load_1m         1-minute load average
#   cpu_load_5m         5-minute load average
#   cpu_load_15m        15-minute load average
#   cpu_usage_pct       Overall CPU busy % (sampled over interval)
#   cpu_freq_mhz        Average CPU frequency in MHz (from scaling_cur_freq)
#   cpu_temp_c          CPU/SoC temperature in °C (highest found)
#   mem_total_mb        Total system memory in MB
#   mem_used_mb         Used system memory in MB (total - available)
#   mem_available_mb    Available system memory in MB
#   mem_used_pct        Memory usage percentage
#   swap_total_mb       Total swap space in MB
#   swap_used_mb        Used swap space in MB
#   gpu_name            GPU device name
#   gpu_util_pct        GPU utilization %
#   gpu_mem_used_mb     GPU memory used in MB
#   gpu_mem_total_mb    GPU memory total in MB
#   gpu_mem_used_pct    GPU memory usage percentage
#   gpu_temp_c          GPU temperature in °C
#   gpu_power_w         GPU power draw in watts
#   gpu_power_limit_w   GPU power limit in watts
#   gpu_clock_mhz       GPU SM clock speed in MHz
#   gpu_mem_clock_mhz   GPU memory clock speed in MHz
#   sparkrun_jobs       Number of running sparkrun_ containers
#   sparkrun_job_names  Pipe-delimited names of running sparkrun_ containers

set -uo pipefail

INTERVAL="${1:-${MONITOR_INTERVAL:-2}}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Snapshot /proc/stat CPU counters (total across all cores).
# Returns: user nice system idle iowait irq softirq steal
read_cpu_stat() {
    awk '/^cpu / {print $2, $3, $4, $5, $6, $7, $8, $9}' /proc/stat
}

# Compute overall CPU busy % from two /proc/stat snapshots.
calc_cpu_pct() {
    local b_user b_nice b_sys b_idle b_iow b_irq b_sirq b_steal
    local a_user a_nice a_sys a_idle a_iow a_irq a_sirq a_steal
    read -r b_user b_nice b_sys b_idle b_iow b_irq b_sirq b_steal <<< "$1"
    read -r a_user a_nice a_sys a_idle a_iow a_irq a_sirq a_steal <<< "$2"

    local b_total=$((b_user + b_nice + b_sys + b_idle + b_iow + b_irq + b_sirq + b_steal))
    local a_total=$((a_user + a_nice + a_sys + a_idle + a_iow + a_irq + a_sirq + a_steal))

    local total_d=$((a_total - b_total))
    local idle_d=$((a_idle - b_idle))

    if [ "$total_d" -le 0 ]; then
        echo "0.0"
        return
    fi

    # Integer math: pct = (total_d - idle_d) * 1000 / total_d, then insert decimal
    local busy_d=$((total_d - idle_d))
    local pct_x10=$(( busy_d * 1000 / total_d ))
    local whole=$((pct_x10 / 10))
    local frac=$((pct_x10 % 10))
    echo "${whole}.${frac}"
}

# Average CPU frequency from sysfs (MHz).  Falls back to /proc/cpuinfo.
read_cpu_freq() {
    local total=0 count=0 freq
    if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
        for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
            [ -r "$f" ] || continue
            freq=$(cat "$f" 2>/dev/null) || continue
            total=$((total + freq))
            count=$((count + 1))
        done
        if [ "$count" -gt 0 ]; then
            echo $(( total / count / 1000 ))
            return
        fi
    fi
    # Fallback: /proc/cpuinfo (some ARM kernels expose "cpu MHz" or
    # "BogoMIPS" but not always MHz — try it anyway)
    local sum
    sum=$(awk '/^cpu MHz/ {s+=$4; n++} END {if(n>0) printf "%.0f", s/n}' /proc/cpuinfo 2>/dev/null)
    if [ -n "$sum" ] && [ "$sum" != "0" ]; then
        echo "$sum"
    else
        echo ""
    fi
}

# Highest CPU/SoC temperature in °C from thermal zones.
read_cpu_temp() {
    local max_temp=0 temp
    for tz in /sys/class/thermal/thermal_zone*/temp; do
        [ -r "$tz" ] || continue
        temp=$(cat "$tz" 2>/dev/null) || continue
        # thermal zone temps are in millidegrees
        if [ "$temp" -gt "$max_temp" ] 2>/dev/null; then
            max_temp=$temp
        fi
    done
    if [ "$max_temp" -gt 0 ]; then
        # Integer division: millideg -> deg with one decimal
        local whole=$((max_temp / 1000))
        local frac=$(( (max_temp % 1000) / 100 ))
        echo "${whole}.${frac}"
    else
        echo ""
    fi
}

# Memory stats from /proc/meminfo (MB).
# Outputs: total_mb used_mb avail_mb pct swap_total_mb swap_used_mb
read_mem() {
    awk '
        /^MemTotal:/     {total=$2}
        /^MemAvailable:/ {avail=$2}
        /^SwapTotal:/    {swap_total=$2}
        /^SwapFree:/     {swap_free=$2}
        END {
            used = total - avail
            total_mb = int(total / 1024)
            avail_mb = int(avail / 1024)
            used_mb  = int(used / 1024)
            pct = (total > 0) ? int(used * 1000 / total) : 0
            whole = int(pct / 10)
            frac  = pct % 10
            swap_total_mb = int(swap_total / 1024)
            swap_used_mb  = int((swap_total - swap_free) / 1024)
            printf "%d %d %d %d.%d %d %d\n", total_mb, used_mb, avail_mb, whole, frac, swap_total_mb, swap_used_mb
        }
    ' /proc/meminfo
}

# GPU metrics via a single nvidia-smi query.
# Returns fields pipe-delimited to avoid ambiguity with commas in GPU names.
read_gpu() {
    nvidia-smi \
        --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 \
    | sed 's/, /|/g'
}

# Trim whitespace and normalize [N/A] to empty string.
clean() {
    echo "$1" | sed 's/^ *//;s/ *$//;s/\[N\/A\]//'
}

# Count running Docker containers whose name starts with "sparkrun_".
count_sparkrun_jobs() {
    docker ps --filter "name=^sparkrun_" --format "{{.ID}}" 2>/dev/null | wc -l | tr -d ' '
}

# List running Docker container names starting with "sparkrun_" (pipe-delimited).
list_sparkrun_job_names() {
    docker ps --filter "name=^sparkrun_" --format "{{.Names}}" 2>/dev/null | tr '\n' '|' | sed 's/|$//'
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

# Take an initial CPU stat snapshot for the first interval.
prev_cpu=$(read_cpu_stat)
# Brief sleep so the first sample has a meaningful CPU delta.
sleep 0.2
cur_cpu=$(read_cpu_stat)
cpu_pct=$(calc_cpu_pct "$prev_cpu" "$cur_cpu")
prev_cpu=$cur_cpu

while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    hn=$(hostname)
    uptime_sec=$(awk '{printf "%.0f", $1}' /proc/uptime)

    # Load averages
    read -r load1 load5 load15 _ _ <<< "$(cat /proc/loadavg)"

    # CPU frequency and temperature
    cpu_freq=$(read_cpu_freq)
    cpu_temp=$(read_cpu_temp)

    # Memory
    read -r mem_total mem_used mem_avail mem_pct swap_total swap_used <<< "$(read_mem)"

    # GPU (single call to nvidia-smi, pipe-delimited)
    gpu_raw=$(read_gpu)
    if [ -n "$gpu_raw" ]; then
        IFS='|' read -r gpu_name gpu_util gpu_mem_used gpu_mem_total gpu_temp gpu_power gpu_power_limit gpu_clock gpu_mem_clock <<< "$gpu_raw"
        gpu_name=$(clean "$gpu_name")
        gpu_util=$(clean "$gpu_util")
        gpu_mem_used=$(clean "$gpu_mem_used")
        gpu_mem_total=$(clean "$gpu_mem_total")
        gpu_temp=$(clean "$gpu_temp")
        gpu_power=$(clean "$gpu_power")
        gpu_power_limit=$(clean "$gpu_power_limit")
        gpu_clock=$(clean "$gpu_clock")
        gpu_mem_clock=$(clean "$gpu_mem_clock")
        # Compute GPU memory usage percentage
        if [ -n "$gpu_mem_used" ] && [ -n "$gpu_mem_total" ] && [ "$gpu_mem_total" != "0" ]; then
            gpu_mem_pct=$(awk "BEGIN {printf \"%.1f\", ($gpu_mem_used/$gpu_mem_total)*100}")
        else
            gpu_mem_pct=""
        fi
    else
        gpu_name=""
        gpu_util=""
        gpu_mem_used=""
        gpu_mem_total=""
        gpu_mem_pct=""
        gpu_temp=""
        gpu_power=""
        gpu_power_limit=""
        gpu_clock=""
        gpu_mem_clock=""
    fi

    # Sparkrun container count and names
    sparkrun_jobs=$(count_sparkrun_jobs)
    sparkrun_job_names=$(list_sparkrun_job_names)

    # Emit CSV row
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$ts" "$hn" "$uptime_sec" \
        "$load1" "$load5" "$load15" \
        "$cpu_pct" "$cpu_freq" "$cpu_temp" \
        "$mem_total" "$mem_used" "$mem_avail" "$mem_pct" \
        "$swap_total" "$swap_used" \
        "$gpu_name" "$gpu_util" "$gpu_mem_used" "$gpu_mem_total" "$gpu_mem_pct" \
        "$gpu_temp" "$gpu_power" "$gpu_power_limit" "$gpu_clock" "$gpu_mem_clock" \
        "$sparkrun_jobs" "$sparkrun_job_names"

    # Sleep, then sample CPU counters for the next iteration's cpu_usage_pct
    sleep "$INTERVAL"

    cur_cpu=$(read_cpu_stat)
    cpu_pct=$(calc_cpu_pct "$prev_cpu" "$cur_cpu")
    prev_cpu=$cur_cpu
done
