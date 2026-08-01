# Low-VRAM training

Reels can reduce persistent transformer memory without quantizing the
trainable LoRA parameters. The low-VRAM path supports per-output-channel
symmetric Int8 storage for frozen dense weights and optional CPU residency:

```toml
[model]
precision = "fp16"

[low_vram]
frozen_weight_quantization = "int8"
cpu_offload = true
```

`frozen_weight_quantization = "int8"` stores every frozen `DenseLayer` and the
base matrix inside every `LoRALinear` as signed Int8 values plus one FP32 scale
per output channel. A projection is dequantized to the activation dtype only
for its matrix multiplication. LoRA A/B matrices, optimizer masters and
moments, normalization parameters, biases, activations, and reductions retain
their normal training dtypes.

Wan preprocessing applies the same setting to the frozen UMT5 encoder. Its
attention and feed-forward matrices use Int8 storage; the token embedding,
relative-position tables, and norms remain in the selected compute dtype.
Programmatic callers can request on-load conversion with
`load_text_encoder(...; precision=:fp16, quantization=:int8)`.
For numerical safety, an FP16+Int8 UMT5 keeps those persistent tensors in
FP16/I8 but promotes its residual stream and projection compute to FP32.  The
full UMT5-XXL acceptance fixture checks that real 512-token Wan conditioning
stays finite; an unpromoted FP16 residual stream overflows in the gated FFN.

Reels can also load a UMT5 file that is already quantized. The portable
`reels-umt5-int8-v1` SafeTensors layout stores each dense matrix under its
ordinary `.weight` key as `I8` and adds a sibling `.weight.scale` FP32 vector.
`quantization=:auto` detects this metadata, while `quantization=:int8` accepts
either a regular checkpoint (quantized during loading) or the prequantized
layout. `write_quantized_umt5` and `load_quantized_umt5_encoder` expose the
format for conversion tools and offline checkpoint preparation.

The full UMT5-XXL acceptance conversion produced a 6,735,282,840-byte file
with 410 tensors (`I8` matrices and `F32` scales). Loading that file with
`precision=:fp16, quantization=:auto` generated positive and negative Wan
contexts bit-identical to quantizing the original checkpoint during loading.

With `cpu_offload = false`, quantized frozen weights remain on the GPU. This
reduces their persistent storage from two FP16 bytes to approximately one byte
per parameter plus row scales. With `cpu_offload = true`, the quantized linear
weights of every transformer block remain on the CPU and are transferred and
dequantized projection-by-projection for forward and backward recomputation.
This minimizes persistent GPU weight memory at the cost of PCIe traffic and
substantially lower throughput.

The current low-VRAM path requires FP16 or FP32 compute. BF16 training uses an
FP32 backward shadow model for operation coverage, which defeats the memory
goal, so the config validator rejects BF16 together with quantization or CPU
offload.

FP8 quantization is not implemented. The `int8` setting is signed
per-output-channel integer quantization and is deliberately named separately
from any future FP8 format.

Activation checkpointing should remain enabled:

```toml
[training]
micro_batch_size = 1
gradient_accumulation = 8
activation_checkpointing = true
checkpoint_interval = 1
```

Quantization changes the frozen forward numerics. Resume therefore assumes the
same low-VRAM settings and base checkpoint. Adapter export is unchanged:
exported files contain only ordinary LoRA A/B tensors and remain compatible
with the documented Wan and LTX consumers.

The small CUDA regression covers both GPU-resident Int8 and Int8 CPU offload:

```sh
REELS_TEST_CUDA=true julia --project=. test/runtests.jl
```

The official LTX-2.3 22B acceptance workload is:

```sh
LTX23_CHECKPOINT=/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
REELS_LOW_VRAM_CPU_OFFLOAD=true \
julia --project=. test/gpu/ltx23_official_low_vram.jl
```

Measured hardware, memory, loss, and elapsed-time results are recorded in
The same settings are available for LTX-2.3 configurations.
