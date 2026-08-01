# Reels.jl

Julia package for preprocessing video datasets and training LoRA adapters for:

- Wan 2.1 T2V and I2V;
- LTX-2.3 T2V.

## Requirements

- Julia 1.11;
- an NVIDIA GPU with a CUDA installation supported by CUDA.jl;
- FFmpeg and FFprobe available on `PATH`;
- model checkpoints referenced by the selected configuration.

Instantiate the environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Configuration

Copy and edit one of these files:

- `configs/wan21_1_3b_t2v_lora.toml`
- `configs/wan21_14b_i2v_lora.toml`
- `configs/ltx23_t2v_lora.toml`

Set checkpoint paths, dataset manifest, cache directory, output directory,
training parameters, and validation prompts in the copied file. Dataset
manifest syntax is described in `docs/src/data/manifest_and_cache.md`.

BF16 is the default CUDA precision. FP16 supports static loss scaling through
`training.loss_scale`. Activation checkpointing is controlled by
`training.activation_checkpointing` and `training.checkpoint_interval`.
Frozen transformer weights may be stored as Int8 by setting
`low_vram.frozen_weight_quantization = "int8"`.

## Preprocessing

```sh
julia --project=. bin/reels preprocess --config path/to/config.toml --device cuda
```

Preprocessing writes cached VAE latents and text-conditioning tensors. I2V
configurations also cache first-frame image conditioning.

## Training

```sh
julia --project=. bin/reels train --config path/to/config.toml --device cuda
```

Resume from a checkpoint:

```sh
julia --project=. bin/reels train \
  --config path/to/config.toml --device cuda \
  --resume path/to/checkpoint.reels
```

The output directory contains:

- `adapter-final.safetensors`;
- periodic adapter and training checkpoints when enabled;
- `metrics.jsonl`;
- `run-summary.json`;
- validation latents when validation is enabled.

## Inspection and validation

```sh
julia --project=. bin/reels inspect path/to/adapter-final.safetensors
julia --project=. bin/reels validate --run path/to/run-directory
```

## Tests

Run CPU tests:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run CUDA tests on an explicitly selected GPU:

```sh
CUDA_VISIBLE_DEVICES=1 REELS_RUN_CUDA_TESTS=true \
  julia --project=. -e 'using Pkg; Pkg.test()'
```

Additional CLI, distributed-training, and low-memory options are documented
under `docs/src/training/`.

## License

Reels is licensed under Apache-2.0. Model checkpoints are distributed
separately and may have additional license terms.
