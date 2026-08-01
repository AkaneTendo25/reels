# Distributed training

Reels supports single-host, one-process-per-GPU data parallel training through
NCCL. Each rank receives a disjoint shard of one deterministic global cache
permutation. LoRA gradients are averaged before clipping and the optimizer
update, so the effective batch size is:

```text
micro_batch_size * gradient_accumulation * world_size
```

Enable the runtime in the training config:

```toml
[distributed]
enabled = true
backend = "nccl"
world_size = 0
rank = -1
local_rank = -1
master_addr = "127.0.0.1"
master_port = 29500
timeout_seconds = 120.0
```

Zero and negative rank fields select the standard launcher environment
variables: `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, and
`MASTER_PORT`. For example, these two concurrent processes train on two GPUs:

```sh
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  julia --project=. bin/reels train --config config.toml --device cuda

CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  julia --project=. bin/reels train --config config.toml --device cuda
```

If each process is isolated to one visible GPU instead, set
`CUDA_VISIBLE_DEVICES` separately and use `LOCAL_RANK=0` in both processes.
All ranks must use the same config, world size, master address, and master
port. Start rank zero first; the other ranks must connect before
`timeout_seconds` expires.

Reels uses the bundled `NCCL_jll` runtime by default. `REELS_NCCL_LIBRARY` may
select another compatible NCCL library, and launchers that already distribute
an NCCL identifier may supply it as 256 hexadecimal characters in
`REELS_NCCL_UNIQUE_ID`.

Only rank zero writes metrics, checkpoints, validation artifacts, run
summaries, and final adapters. All ranks participate in the training and final
barrier. Cache sampling requires at least
`micro_batch_size * world_size` usable entries in each sampled batch.

Resume is supported with the same world size that created the checkpoint.
Reels records this value in checkpoint metadata and rejects a changed process
count, because changing it would alter the data/RNG trajectory and cannot
provide exact resume semantics.

The two-rank CUDA integration fixture exercises real NCCL communication,
rank-safe artifacts, and comparison against a single-GPU effective-batch
baseline:

```sh
bash test/gpu/run_distributed_test.sh
```
