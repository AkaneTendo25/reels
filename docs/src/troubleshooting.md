# Troubleshooting

## CUDA is unavailable

Confirm that CUDA.jl sees the intended device and that `CUDA_VISIBLE_DEVICES`
maps it correctly. Distributed training uses one process per GPU and
`LOCAL_RANK` refers to the visible-device index, not necessarily the physical
index.

## Out of memory

Use FP16 for large official models, keep activation checkpointing enabled,
reduce the frame/resolution bucket or micro-batch size, and increase gradient
accumulation to preserve effective batch. Then enable Int8 frozen weights and,
if required, CPU offload. CPU offload trades substantial throughput for lower
persistent device memory.

Trainer OOM errors include the active latent-bucket shape and dtype, activation
checkpoint settings, and current Int8/offload mode. The message also lists the
safe memory levers, so an allocation failure is not mistaken for a checkpoint
or cache-format error.

## NCCL initialization fails or hangs

Verify identical `WORLD_SIZE`, unique `RANK` values, correct `LOCAL_RANK`
values, and one shared `MASTER_ADDR`/`MASTER_PORT`. Reels uses `NCCL_jll` by
default; an incompatible system NCCL should not be forced through
`REELS_NCCL_LIBRARY`.

## Checkpoint or adapter keys do not match

Run `bin/reels inspect` and check the model family, base checkpoint, adapter
format, rank, and mapping prefix. Reels deliberately rejects missing,
unexpected, or shape-mismatched tensors instead of partially loading them.

## Cache entries are rejected

Re-run preprocessing after changing checkpoint fingerprints, tokenizer/VAE
identity, dtype, target FPS, frame buckets, resolution, crop behavior, or cache
schema. Cache identities intentionally include these inputs.

## SentencePiece is unavailable

Run `julia --project=. deps/build.jl`. If the bridge is already built
elsewhere, set `REELS_SENTENCEPIECE_LIBRARY` to its absolute shared-library
path. Do not point it at the raw SentencePiece C++ library; Reels expects its
own bridge ABI.

## Non-finite loss or gradients

Reels stops before the optimizer update and reports the affected step. Inspect
the cache for non-finite tensors, lower the learning rate, verify the compute
precision, and reproduce with FP32 on a tiny fixture. Do not weaken global
parity tolerances to hide the failure.
