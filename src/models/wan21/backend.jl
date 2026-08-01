struct Wan21 <: AbstractVideoModelBackend
    variant::Symbol
    checkpoint::String
end
Wan21(; variant=:t2v_1_3b, checkpoint="") = Wan21(variant, String(checkpoint))

model_family(::Wan21) = :wan21
load_config(backend::Wan21, source=nothing) = wan21_config(backend.variant)

struct WanTextConditioner
    encoder::UMT5Encoder
    tokenizer::SentencePieceTokenizer
    max_length::Int
    device::Symbol
end

struct WanTextEncoding{C<:AbstractArray,M<:AbstractMatrix}
    context::C
    mask::M
    lengths::Vector{Int}
end

_tokenizer_vocabulary_fits(tokenizer_size::Integer,
                           embedding_size::Integer) =
    0 < tokenizer_size <= embedding_size

function load_transformer(backend::Wan21, checkpoint=backend.checkpoint,
                          device=:cpu, precision=:fp32)
    device in (:cpu, :cuda) ||
        throw(ArgumentError("Wan device must be :cpu or :cuda"))
    precision in (:fp32, :bf16, :fp16) ||
        throw(ArgumentError("unsupported Wan transformer precision $precision"))
    isempty(checkpoint) && throw(ArgumentError("Wan checkpoint path is empty"))
    model = load_wan_transformer(checkpoint, wan21_config(backend.variant))
    move_to_device(model, device, precision)
end

"""
Load a native Wan UMT5 text conditioner from a SafeTensors encoder checkpoint
and the official UMT5 SentencePiece model. The original Wan and official
Diffusers repositories expose the same 242-key encoder state dictionary.
"""
function load_text_encoder(::Wan21, checkpoint::AbstractString;
                           tokenizer_model::AbstractString,
                           device=:cpu, precision=:fp32,
                           config=UMT5Config(), strict=true,
                           quantization=:none, cpu_offload=false)
    device in (:cpu, :cuda) ||
        throw(ArgumentError("Wan text encoder device must be :cpu or :cuda"))
    precision in (:fp32, :bf16, :fp16) ||
        throw(ArgumentError("unsupported Wan text encoder precision $precision"))
    quantization in (:none, :int8, :auto) || throw(ArgumentError(
        "Wan text encoder quantization must be :none, :int8, or :auto"))
    endswith(lowercase(checkpoint), ".pth") &&
        throw(ArgumentError("native UMT5 runtime accepts SafeTensors, not PyTorch " *
            ".pth; use the official Wan Diffusers text_encoder directory"))
    source = open_tensor_source(checkpoint)
    metadata = source isa SingleTensorSource ? source.header.metadata :
        Dict{String,String}()
    prequantized = get(metadata, "format", "") == UMT5_INT8_FORMAT
    requested = quantization === :auto ?
        (prequantized ? :int8 : :none) : quantization
    prequantized && requested !== :int8 && throw(ArgumentError(
        "quantized UMT5 checkpoint requires quantization=:int8 or :auto"))
    (requested === :int8 || cpu_offload) && precision === :bf16 &&
        throw(ArgumentError(
            "Int8/offloaded UMT5 currently requires fp16 or fp32 compute"))
    T = precision_eltype(precision)
    encoder = prequantized ?
        load_quantized_umt5_encoder(source, config;
            compute_type=T, strict=strict) :
        load_umt5_encoder(source, config; strict=strict)
    transfer = requested === :int8 || cpu_offload ?
        frozen_weight_transfer(device, precision;
            quantization=requested, cpu_offload=cpu_offload) :
        array_transfer(device, precision)
    encoder = move_to_device(encoder, transfer)
    tokenizer = SentencePieceTokenizer(tokenizer_model)
    # Official UMT5 pads its 256,000-piece SentencePiece vocabulary to a
    # 256,384-row embedding matrix for accelerator-friendly dimensions.
    _tokenizer_vocabulary_fits(tokenizer.vocab_size, config.vocab_size) ||
        throw(ArgumentError("tokenizer vocabulary $(tokenizer.vocab_size) " *
            "exceeds encoder embedding vocabulary $(config.vocab_size)"))
    WanTextConditioner(encoder, tokenizer, 512, device)
end

function load_vae(::Wan21, checkpoint::AbstractString;
                  device=:cpu, precision=:fp32,
                  config=WanVAEConfig())
    device in (:cpu, :cuda) ||
        throw(ArgumentError("Wan VAE device must be :cpu or :cuda"))
    precision in (:fp32, :bf16, :fp16) ||
        throw(ArgumentError("unsupported Wan VAE precision $precision"))
    endswith(lowercase(checkpoint), ".pth") &&
        throw(ArgumentError("native Wan VAE runtime accepts SafeTensors, not " *
            "PyTorch .pth; use the official Wan Diffusers VAE SafeTensors file"))
    model = load_wan_vae(checkpoint, config)
    move_to_device(model, device, precision)
end

function encode_text(::Wan21, conditioner::WanTextConditioner,
                     texts::AbstractVector{<:AbstractString})
    tokenized = tokenize_wan(conditioner.tokenizer, texts;
        max_length=conditioner.max_length)
    mask = conditioner.device === :cuda ?
        CUDA.CuArray(tokenized.mask) : tokenized.mask
    context = umt5_forward(conditioner.encoder, tokenized.ids; mask=mask)
    # Wan pads contexts with exact zeros after each caption's EOS before
    # cross-attention, rather than retaining encoder activations at pad tokens.
    expanded_mask = reshape(mask, 1, size(mask, 1), size(mask, 2))
    context = if eltype(context) === BFloat16
        bfloat16_values(ifelse.(
            expanded_mask, float32_values(context), 0f0))
    else
        ifelse.(expanded_mask, context, zero(eltype(context)))
    end
    WanTextEncoding(context, mask, tokenized.lengths)
end

encode_text(backend::Wan21, conditioner::WanTextConditioner,
            text::AbstractString) =
    encode_text(backend, conditioner, [text])

function prepare_conditioning(backend::Wan21,
                              conditioner::WanTextConditioner,
                              texts::AbstractVector{<:AbstractString})
    encoded = encode_text(backend, conditioner, texts)
    encoded.context
end

prepare_conditioning(backend::Wan21, conditioner::WanTextConditioner,
                     text::AbstractString) =
    dropdims(prepare_conditioning(backend, conditioner, [text]); dims=3)

function encode_video(::Wan21, vae::WanVAEEncoder, video::AbstractArray)
    ndims(video) in (4, 5) ||
        throw(DimensionMismatch("video must be (C,T,H,W) or (C,T,H,W,B)"))
    batched = ndims(video) == 4 ? reshape(video, size(video)..., 1) : video
    on_cuda = vae.input_conv.weight isa CUDA.CuArray
    input = on_cuda && !(batched isa CUDA.CuArray) ?
        CUDA.CuArray(batched) : batched
    encoded = wan_vae_encoder_forward(vae, input)
    ndims(video) == 4 ? dropdims(encoded; dims=5) : encoded
end

encode_video(backend::Wan21, vae::WanVAE, video::AbstractArray) =
    encode_video(backend, vae.encoder, video)

"""
    prepare_wan_i2v_conditioning(backend, vae, video, image_features)

Construct the official Wan 2.1 I2V training inputs from a decoded clip. The
target clip is VAE encoded normally. A second clip containing only the first
frame is encoded and concatenated with the four-channel temporal mask used by
the official implementation. `image_features` are the 1280-channel CLIP vision
tokens for that same first frame.
"""
function prepare_wan_i2v_conditioning(backend::Wan21,
                                      vae::Union{WanVAEEncoder,WanVAE},
                                      video::AbstractArray,
                                      image_features::AbstractMatrix;
                                      target_latents=nothing)
    config = wan21_config(backend.variant)
    config.model_type === :i2v ||
        throw(ArgumentError(
            "Wan I2V conditioning requires an I2V backend variant"))
    ndims(video) == 4 ||
        throw(DimensionMismatch("I2V video must be (C,T,H,W)"))
    size(video, 1) == 3 ||
        throw(DimensionMismatch("I2V video must contain three RGB channels"))
    size(video, 2) > 0 ||
        throw(DimensionMismatch("I2V video must contain at least one frame"))
    size(image_features, 1) == 1280 ||
        throw(DimensionMismatch(
            "Wan I2V image features must have 1280 channels"))
    size(image_features, 2) > 0 ||
        throw(DimensionMismatch(
            "Wan I2V image features must contain at least one token"))

    latents = target_latents === nothing ?
        encode_video(backend, vae, video) : target_latents
    ndims(latents) == 4 && size(latents, 1) == 16 ||
        throw(DimensionMismatch(
            "Wan I2V target latents must be (16,T,H,W)"))
    reference = similar(video)
    fill!(reference, zero(eltype(reference)))
    @views reference[:, 1, :, :] .= video[:, 1, :, :]
    reference_latents = encode_video(backend, vae, reference)
    size(reference_latents) == size(latents) ||
        throw(DimensionMismatch(
            "I2V reference and target VAE latents have different shapes"))

    # The upstream mask is [1,0,...] in video time, with its first value
    # repeated four times before grouping into four latent mask channels.
    mask = similar(reference_latents, eltype(reference_latents),
                   4, size(reference_latents, 2),
                   size(reference_latents, 3), size(reference_latents, 4))
    fill!(mask, zero(eltype(mask)))
    @views mask[:, 1, :, :] .= one(eltype(mask))
    conditioning_video = cat(mask, reference_latents; dims=1)
    (latents=latents, conditioning_video=conditioning_video,
     image_features=image_features)
end

function decode_video(::Wan21, vae::WanVAEDecoder, latents::AbstractArray)
    ndims(latents) in (4, 5) ||
        throw(DimensionMismatch("latents must be (C,T,H,W) or (C,T,H,W,B)"))
    batched = ndims(latents) == 4 ?
        reshape(latents, size(latents)..., 1) : latents
    on_cuda = vae.input_conv.weight isa CUDA.CuArray
    input = on_cuda && !(batched isa CUDA.CuArray) ?
        CUDA.CuArray(batched) : batched
    decoded = wan_vae_decoder_forward(vae, input)
    ndims(latents) == 4 ? dropdims(decoded; dims=5) : decoded
end

decode_video(backend::Wan21, vae::WanVAE, latents::AbstractArray) =
    decode_video(backend, vae.decoder, latents)

function lora_targets(::Wan21, model=nothing, selection=:attention)
    selection === :attention && return [
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o|k_img|v_img)\.weight"]
    selection === :attention_and_ffn && return [
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o|k_img|v_img)\.weight",
        r"blocks\.\d+\.ffn\.(0|2)\.weight"]
    selection === :all_linear && return [r".*\.weight"]
    selection isa AbstractVector && return selection
    throw(ArgumentError("unknown Wan LoRA target preset $selection"))
end

export_mapping(::Wan21) = Dict(
    "lora_A" => "lora_A.weight",
    "lora_B" => "lora_B.weight",
)
