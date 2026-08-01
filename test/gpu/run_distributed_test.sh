#!/usr/bin/env bash
set -u

mapfile -t physical_gpus < <(
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
  awk -F, '$1+0 != 0 && $2+0 < 512 && $3+0 < 5 {
    gsub(/ /, "", $1); print $1
  }' |
  head -2
)

if [[ ${#physical_gpus[@]} -ne 2 ]]; then
  echo "need two idle physical GPUs other than GPU 0" >&2
  exit 2
fi

output_dir=${1:-$(mktemp -d "${TMPDIR:-/tmp}/reels-nccl-test.XXXXXX")}
mkdir -p "$output_dir"
master_port=${MASTER_PORT:-$((30000 + $$ % 10000))}
julia_bin=${JULIA_BIN:-$(command -v julia)}

export CUDA_VISIBLE_DEVICES="${physical_gpus[0]},${physical_gpus[1]}"
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$master_port"

echo "PHYSICAL_GPUS=${physical_gpus[0]},${physical_gpus[1]}"
echo "OUTPUT_DIR=$output_dir"

RANK=0 LOCAL_RANK=0 bash -c \
  'exec -a reels-unit-test "$@"' _ \
  "$julia_bin" --project=. test/gpu/distributed_worker.jl "$output_dir" \
  >"$output_dir/rank-0.log" 2>&1 &
pid0=$!
RANK=1 LOCAL_RANK=1 bash -c \
  'exec -a reels-unit-test "$@"' _ \
  "$julia_bin" --project=. test/gpu/distributed_worker.jl "$output_dir" \
  >"$output_dir/rank-1.log" 2>&1 &
pid1=$!

set +e
wait "$pid0"
status0=$?
wait "$pid1"
status1=$?
set -e

echo "RANK_0_STATUS=$status0"
echo "RANK_1_STATUS=$status1"
sed 's/^/[rank 0] /' "$output_dir/rank-0.log"
sed 's/^/[rank 1] /' "$output_dir/rank-1.log"

if [[ $status0 -ne 0 || $status1 -ne 0 ]]; then
  exit 1
fi

bash -c 'exec -a reels-unit-test "$@"' _ \
  "$julia_bin" --project=. test/gpu/distributed_check.jl "$output_dir"
