# Resume and recovery

Every `.reels` checkpoint contains the optimizer step, micro-step, Xoshiro RNG
state, trainable LoRA tensors, FP32 AdamW moments/masters, and versioned
identity metadata. Adapter snapshots are SafeTensors and are not sufficient to
resume optimizer state.

Resume from an optimizer-step boundary:

```sh
julia --project=. bin/reels train \
  --config config.toml --device cuda \
  --resume runs/example/checkpoint-250.reels
```

Reels rejects a mismatched model family, base checkpoint, distributed world
size, frozen-weight quantization mode, or CPU-offload setting. Exact
distributed resume is supported with the same world size. Older checkpoints
without newer identity fields remain readable but cannot prove those omitted
identities.

Only rank zero writes checkpoints. Files are first written to temporary paths
and atomically renamed. `checkpoint.keep_last` prunes older paired checkpoint
and adapter snapshots only after a new one is complete.

When resuming into an existing output directory, metrics newer than the
restored step are removed before appending. Use:

```sh
julia --project=. bin/reels validate --run runs/example
```

to audit the final adapter, checkpoints, metrics, summary, and validation
artifacts. If a process died during a write, ignore orphan `.tmp` files and
resume from the newest complete `.reels` checkpoint.
