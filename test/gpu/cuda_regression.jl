using Test
using Random
using Reels
using CUDA

Sys.islinux() &&
    ccall(:prctl, Cint, (Cint, Cstring, Culong, Culong, Culong),
          15, "reels-unit-test", 0, 0, 0)

length(CUDA.devices()) == 1 ||
    error("CUDA regression runner requires exactly one visible GPU")
CUDA.device!(0)

default_files = ["lora.jl", "low_vram.jl", "ltx23.jl", "wan_overfit.jl"]
allowed_files = Set(default_files)
selected_files = isempty(ARGS) ? default_files : ARGS
all(file -> file in allowed_files, selected_files) ||
    error("unknown CUDA regression file in $(join(selected_files, ", "))")
foreach(include, selected_files)
