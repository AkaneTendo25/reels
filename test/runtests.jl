using Reels
using Random
using Test

include("unit/config.jl")
include("unit/public_api.jl")
include("unit/quality.jl")
include("unit/data.jl")
include("unit/media.jl")
include("unit/safetensors.jl")
include("unit/lora.jl")
include("unit/quantization.jl")
include("unit/layers.jl")
include("unit/state_dict.jl")
include("unit/wan21.jl")
include("unit/wan_text_encoder.jl")
include("unit/wan_vae.jl")
include("unit/wan_image_encoder.jl")
include("unit/wan_block.jl")
include("unit/wan_transformer.jl")
include("unit/wan_lora.jl")
include("unit/wan_training.jl")
include("unit/wan_runner.jl")
include("unit/ltx23.jl")
include("unit/ltx23_vae.jl")
include("unit/ltx23_gemma.jl")
include("unit/ltx23_text_connector.jl")
include("unit/ltx23_runner.jl")
include("unit/training.jl")
include("unit/cli.jl")
include("parity/wan_block.jl")
include("parity/wan_vae_encoder.jl")
include("parity/umt5_tokenizer.jl")

if get(ENV, "REELS_TEST_CUDA", "false") == "true"
    include("gpu/lora.jl")
    include("gpu/low_vram.jl")
    include("gpu/ltx23.jl")
    include("gpu/wan_overfit.jl")
    isempty(get(ENV, "WAN21_1_3B_CHECKPOINT", "")) ||
        include("gpu/wan21_official_training.jl")
end
