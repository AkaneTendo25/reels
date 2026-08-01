# Command-line workflow

The CLI and Julia API consume the same strict TOML configuration.

```sh
julia --project=. bin/reels preprocess \
  --config configs/wan21_1_3b_t2v_lora.toml --device cuda

julia --project=. bin/reels preprocess \
  --config configs/wan21_14b_i2v_lora.toml --device cuda

julia --project=. bin/reels preprocess \
  --config configs/ltx23_t2v_lora.toml --device cuda

julia --project=. bin/reels train \
  --config configs/wan21_1_3b_t2v_lora.toml --device cuda

julia --project=. bin/reels train \
  --config configs/wan21_1_3b_t2v_lora.toml --device cuda \
  --resume runs/wan21-1.3b-lora/checkpoint-250.reels

julia --project=. bin/reels inspect \
  runs/wan21-1.3b-lora/adapter-final.safetensors

julia --project=. bin/reels validate --run runs/wan21-1.3b-lora
```

For multi-GPU training, enable `[distributed]` and launch one CLI process per
GPU. See [Distributed training](distributed.md) for rank variables, effective
batch sizing, rank-zero artifact behavior, and resume constraints.

For Int8 frozen weights or CPU-resident transformer block weights, see
[Low-VRAM training](low_vram.md). These modes use the same `train` command and
are selected only through `[low_vram]`.

For Wan T2V, `preprocess` computes native UMT5 context and causal-VAE latents.
For Wan I2V it additionally loads the native CLIP ViT-H/14 encoder, encodes
the first frame into 257 image tokens, and constructs the official
four-channel mask plus first-frame VAE conditioning. Set
`model.image_encoder_checkpoint` to the official image-encoder SafeTensors
file or directory. For
LTX-2.3, it computes native Gemma-3 hidden states, the LTX feature projection
and connector, causal video-VAE latents, latent patches, and official
time/height/width midpoint coordinates. Both paths write content-addressed
atomic SafeTensors entries and reuse existing valid entries. `train` loads only
the diffusion transformer and cached entries, injects the configured LoRA
preset, and performs resumable training.

When `[validation]` is enabled, preprocessing also encodes every validation
prompt and creates deterministic fixed-noise cache entries using the run seed.
Training therefore does not reload the text encoder or VAE. At each configured
interval it runs the native rectified-flow Euler sampler twice—once with the
adapter enabled and once disabled—and writes both latent outputs under
`OUTPUT/validation/step-N/`. Each artifact records the prompt, step, and mean
absolute adapter delta.

```toml
[validation]
every_steps = 250
inference_steps = 20
prompts = ["A red kite rises above a windy beach."]
```

The LTX text encoder path accepts a Gemma directory containing `config.json`
and one `*.safetensors.index.json`; `model.tokenizer_model` points to its
SentencePiece model. The LTX transformer checkpoint supplies the native video
connector, and `model.vae_checkpoint` supplies the video VAE.

The default CUDA compute path is BF16. Frozen transformer parameters,
activations, and cached batches use BF16; normalization, attention statistics,
loss reductions, LoRA gradients, AdamW moments, and optimizer master parameters
use FP32 stability boundaries. Reels uses bit-exact BF16 conversion kernels
because direct `BFloat16.(CuArray(...))` lowering is incorrect on affected
Julia/PTX combinations.

BF16 execution is CUDA-only. Use `precision = "fp32"` for CPU diagnostics.
FP16 configurations may set `training.loss_scale` to a positive static scale.
Reels unscales into FP32 before accumulation, NCCL reduction, clipping, and the
optimizer update; the reported loss and gradient norm remain unscaled.
Activation checkpointing is enabled by default at every transformer block and
can be adjusted with `training.checkpoint_interval`.

Every training run writes `metrics.jsonl`, with one versioned record per
optimizer step containing raw and EMA-smoothed loss, learning rate, gradient
and adapter norms, the active latent-bucket shape, and
data/train/optimizer/step timings. CUDA records additionally include effective,
pool-used, pool-reserved, and run-peak device-memory bytes. Completion
atomically writes
`run-summary.json`, which records the final step, adapter, validation count,
and latest metric. When resuming in an existing output directory, records
newer than the restored checkpoint are removed before training continues.
`reels validate --run` audits these files when present in addition to adapters,
checkpoints, and validation artifacts.

Resume validates the optimizer-affecting training configuration, flow-matching
settings, distributed world size, low-VRAM mode, and the exact ordered adapter
layout (path, rank, alpha, dropout, trainable-bias flag, and tensor shapes).
Incompatible runs stop before parameter restoration and name the mismatched
configuration key. Extending a constant-schedule run is allowed; changing the
final step of a linear or cosine schedule is rejected because it changes the
next learning rate.

LTX configs use the official shifted logit-normal flow-matching timestep
sampler:

```toml
[flow_matching]
timestep_sampling = "shifted_logit_normal"
standard_deviation = 1.0
epsilon = 0.001
uniform_probability = 0.1
```
