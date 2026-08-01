# Configuration reference

Reels reads strict TOML with `schema_version = 1`. Unknown fields are rejected.
Paths are data and are never executed.

## Top level

- `seed`: deterministic preprocessing/training seed.
- `output_dir`: run artifacts.
- `[model]`: family, variant, checkpoint paths, and compute precision.
- `[data]`: manifest, cache, buckets, FPS, and preprocessing workers.
- `[adapter]`: LoRA rank, alpha, dropout, and target preset.
- `[training]`: steps, batch/accumulation, optimizer, clipping, and activation
  checkpointing.
- `[checkpoint]`, `[validation]`, `[flow_matching]`, `[distributed]`, and
  `[low_vram]`: the corresponding runtime features.

## Model

`family` is `wan21` or `ltx23`; `variant` selects a supported backend variant.
`checkpoint` is the diffusion transformer SafeTensors file or directory.
Wan preprocessing also uses `text_encoder_checkpoint`, `tokenizer_model`, and
`vae_checkpoint`; I2V additionally requires `image_encoder_checkpoint`. LTX
uses a Gemma directory, tokenizer model, and the LTX VAE checkpoint.

`precision` is `bf16`, `fp16`, or `fp32`. CPU diagnostics use FP32. Low-VRAM
quantization/offload currently requires FP16 or FP32. For Wan preprocessing,
`low_vram.frozen_weight_quantization = "int8"` also quantizes frozen UMT5
dense weights; a prequantized `reels-umt5-int8-v1` text checkpoint is loaded
without reconstructing full-precision dense matrices.

## Data

`manifest` points to the strict dataset TOML/JSONL manifest and `cache_dir`
holds content-addressed preprocessing entries. `frame_buckets` are compatible
frame counts and `resolution_buckets` use `WIDTHxHEIGHT`. `target_fps` controls
deterministic temporal sampling.

## Adapter and training

`targets` is a model-specific preset such as `attention`. Effective global
batch size is:

```text
micro_batch_size * gradient_accumulation * distributed_world_size
```

Adapter `dropout` uses one deterministic mask stream per adapted projection and
one seed per micro-step. Activation-checkpoint recomputation reuses the same
mask, and the training RNG checkpoint makes masks reproducible after resume.
Validation and adapter-disabled comparisons always run without dropout.

`train_bias = true` copies and trains the bias of every targeted projection;
those tensors are included in optimizer state, checkpoints, and adapter
exports. Exact per-module rank and alpha settings accept either the official
module path or its `.weight` form:

```toml
[adapter.rank_overrides]
"blocks.0.self_attn.q" = 8

[adapter.alpha_overrides]
"blocks.0.self_attn.q.weight" = 16.0
```

An override that does not identify an injected target is rejected.

AdamW uses `learning_rate`, `weight_decay`, and `max_gradient_norm`. FP16 runs
may set a positive static `loss_scale`; the backward objective is scaled and
gradients are converted to FP32 and unscaled before accumulation, distributed
reduction, clipping, and AdamW. Other precisions require `loss_scale = 1`.
`weight_decay_exclusions` is an optional list of Julia regular expressions
matched against stable adapter parameter names; for example,
`weight_decay_exclusions = ["bias$"]` keeps trainable biases out of decay.
`activation_checkpointing = true` with `checkpoint_interval = 1` is the
production memory default. Scheduler `type` is `constant` or `linear`, with
`cosine` also supported, and accepts optional `warmup_steps`.

## Checkpoint, validation, and flow

`checkpoint.every_steps` and `keep_last` control optimizer/RNG snapshots and
adapter snapshots. Validation prompts are preprocessed into deterministic
conditioning caches; `every_steps` and `inference_steps` control comparisons.

Flow matching supports `auto`, `uniform`, and `shifted_logit_normal` timestep
sampling. LTX defaults to shifted logit-normal; Wan defaults to uniform.

## Distributed and low-VRAM

See [Distributed training](training/distributed.md) for NCCL rank settings and
[Low-VRAM training](training/low_vram.md) for Int8/offload behavior.

The checked examples under `configs/` are the authoritative complete starting
points and are parsed by the test suite.
