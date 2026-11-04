#!/usr/bin/env bash
set -u

# Runs benchmark_batch.py on the RTX 4090 with the other services on that GPU
# stopped. The containers are restarted when the script exits for any reason
# (success, failure, Ctrl+C or kill), and the script waits until they report
# healthy again. Arguments are passed through to benchmark_batch.py.
#
# Run from anywhere. If the machine reboots mid-run, start them manually:
#   docker start slm_serve_engine_v2 omnivoicecpp_triton_test

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIOCPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$AUDIOCPP_ROOT"

# Containers that use the RTX 4090 (nvidia-smi index 0).
CONTAINERS=(slm_serve_engine_v2 omnivoicecpp_triton_test)

OUTPUT_DIR="omnivoice_python_cpp_so/outputs/benchmark_batch"
LOG="$OUTPUT_DIR/benchmark_batch.log"

# Containers this script stopped. Kept outside OUTPUT_DIR and removed only
# after they are restarted, so a run killed with SIGKILL (no trap) leaves the
# list behind and the next run restarts them too.
STATE="omnivoice_python_cpp_so/outputs/.benchmark_stopped_containers"

mkdir -p "$OUTPUT_DIR"
: > "$LOG"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [wrapper] $*" | tee -a "$LOG"
}

STOPPED=()

if [ -f "$STATE" ]; then
    while read -r name; do
        [ -n "$name" ] || continue
        log "$name was stopped by an earlier run, will restart it at the end"
        STOPPED+=("$name")
    done < "$STATE"
fi

restart_containers() {
    trap - EXIT INT TERM

    for name in "${STOPPED[@]}"; do
        log "starting $name"
        docker start "$name" >> "$LOG" 2>&1 || log "ERROR: docker start $name failed"
    done

    # Wait up to 15 minutes for each restarted container to become healthy.
    for name in "${STOPPED[@]}"; do
        for _ in $(seq 1 180); do
            health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$name" 2>/dev/null)"
            if [ "$health" = "healthy" ] || [ "$health" = "running" ]; then
                break
            fi
            sleep 5
        done
        log "$name: $health"
    done

    rm -f "$STATE"

    log "GPU 0 after restart:"
    nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG" 2>&1
}

trap restart_containers EXIT
trap 'log "interrupted"; exit 130' INT TERM

for name in "${CONTAINERS[@]}"; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = "true" ]; then
        log "stopping $name"
        STOPPED+=("$name")
        printf '%s\n' "${STOPPED[@]}" | sort -u > "$STATE"
        docker stop "$name" >> "$LOG" 2>&1 || log "ERROR: docker stop $name failed"
    elif [[ ! " ${STOPPED[*]} " =~ " $name " ]]; then
        log "$name is not running, leaving it alone"
    fi
done

log "GPU 0 before benchmark:"
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv >> "$LOG" 2>&1

python3 omnivoice_python_cpp_so/benchmark_batch.py "$@"
status=$?

log "benchmark exit code $status"
exit $status
