# Installation and CUDA prerequisites

Reels requires Julia 1.11. Clone the repository, instantiate the checked-in
environment, and build the native SentencePiece bridge:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. deps/build.jl
julia --project=. -e 'using Reels; println("Reels loaded")'
```

Training requires an NVIDIA GPU supported by CUDA.jl 6.1. The production gates
use CUDA GPUs. BF16 compute requires Ampere-class or newer hardware; FP16 is the
documented large-checkpoint compatibility mode. CUDA, cuDNN, and the pinned
NCCL artifact are resolved by Julia packages; a Python or PyTorch environment
is not required.

FFmpeg and FFprobe must be available on `PATH` for media preprocessing. The
SentencePiece build requires a C++ compiler and SentencePiece development
headers/libraries. If the bridge is installed elsewhere, set
`REELS_SENTENCEPIECE_LIBRARY` to the built shared library.

Run the CPU suite with:

```sh
julia --project=. test/runtests.jl
```

Run opt-in CUDA tests only on an available device:

```sh
CUDA_VISIBLE_DEVICES=0 REELS_TEST_CUDA=true \
  julia --project=. test/runtests.jl
```

Large official-checkpoint tests are separate and require their documented
environment-variable paths. The example commands above are development
commands; release packaging remains subject to the licensing gate in
[Licensing and attribution](licensing.md).
