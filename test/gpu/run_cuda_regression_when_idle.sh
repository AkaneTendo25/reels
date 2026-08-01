#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

julia_bin="${JULIA_BIN:-julia}"

idle_nonzero_gpu() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, '
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        gsub(/[[:space:]]/, "", $3)
      }
      ($1 + 0) != 0 && ($2 + 0) < 512 && ($3 + 0) < 5 {
        print $1
        exit
      }
    '
}

selected_gpu=""
test_pid=""

wait_for_idle_gpu() {
  local candidate=""
  local current_candidate=""
  local stable_samples=0
  while (( stable_samples < 6 )); do
    current_candidate="$(idle_nonzero_gpu)"
    if [[ -n "$current_candidate" && "$current_candidate" == "$candidate" ]]; then
      stable_samples=$((stable_samples + 1))
    elif [[ -n "$current_candidate" ]]; then
      candidate="$current_candidate"
      stable_samples=1
    else
      candidate=""
      stable_samples=0
    fi
    (( stable_samples >= 6 )) && break
    echo "Waiting for an idle physical GPU other than GPU 0"
    sleep 5
  done
  selected_gpu="$candidate"
}

terminate_test() {
  [[ -z "$test_pid" ]] || kill "$test_pid" 2>/dev/null || true
}
trap 'terminate_test; exit 130' INT TERM

if (( $# > 0 )); then
  test_files=("$@")
else
  test_files=("lora.jl" "low_vram.jl" "ltx23.jl" "wan_overfit.jl")
fi
for test_file in "${test_files[@]}"; do
  while true; do
    wait_for_idle_gpu
    candidate="$selected_gpu"
    echo "Using physical GPU $candidate for $test_file"
    CUDA_VISIBLE_DEVICES="$candidate" \
      bash -c \
        'exec -a reels-unit-test "$1" --project=. \
          test/gpu/process_entry.jl test/gpu/cuda_regression.jl "$2"' \
      _ "$julia_bin" "$test_file" &
    test_pid=$!
    interrupted=false

    while kill -0 "$test_pid" 2>/dev/null; do
      foreign_pids="$(
        nvidia-smi -i "$candidate" \
          --query-compute-apps=pid \
          --format=csv,noheader,nounits |
          awk -v own_pid="$test_pid" '
            {
              gsub(/[[:space:]]/, "", $1)
            }
            ($1 + 0) != (own_pid + 0) {
              print $1
            }
          '
      )"
      if [[ -n "$foreign_pids" ]]; then
        echo "Foreign GPU process appeared on physical GPU $candidate; retrying $test_file"
        terminate_test
        wait "$test_pid" 2>/dev/null || true
        test_pid=""
        interrupted=true
        break
      fi
      sleep 5
    done

    "$interrupted" && continue
    set +e
    wait "$test_pid"
    status=$?
    set -e
    test_pid=""
    (( status == 0 )) || exit "$status"
    echo "Completed $test_file"
    break
  done
done
