# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

struct LTX23 <: AbstractVideoModelBackend
    variant::Symbol
    checkpoint::String
    video_only::Bool
end

LTX23(; variant=:av_22b_dev, checkpoint="", video_only=false) =
    LTX23(Symbol(variant), String(checkpoint), Bool(video_only))

model_family(::LTX23) = :ltx23

function load_config(backend::LTX23, source=nothing)
    if source === nothing
        isempty(backend.checkpoint) ?
            LTX23Config(variant=backend.variant, video_only=backend.video_only) :
            ltx23_config(backend.checkpoint; video_only=backend.video_only)
    elseif source isa AbstractDict
        ltx23_config(source; video_only=backend.video_only)
    else
        ltx23_config(String(source); video_only=backend.video_only)
    end
end

function load_transformer(backend::LTX23,
                          checkpoint=backend.checkpoint,
                          device=:cpu, precision=:fp32)
    backend.video_only ||
        throw(ArgumentError("native LTX audio-video transformer construction " *
                            "is not implemented yet; set video_only=true"))
    isempty(checkpoint) &&
        throw(ArgumentError("LTX checkpoint path is empty"))
    config = ltx23_config(checkpoint; video_only=true)
    keys = tensor_keys(open_tensor_source(checkpoint))
    prefix = any(key -> startswith(key, "model.diffusion_model."), keys) ?
        "model.diffusion_model." : ""
    model = load_ltx23_video_transformer(
        checkpoint, config; strict=isempty(prefix),
        checkpoint_prefix=prefix)
    move_to_device(model, device, precision)
end

function load_vae(backend::LTX23, checkpoint=backend.checkpoint,
                  device=:cpu, precision=:fp32)
    isempty(checkpoint) &&
        throw(ArgumentError("LTX checkpoint path is empty"))
    model = load_ltx23_vae_encoder(checkpoint)
    move_to_device(model, device, precision)
end

function load_text_encoder(backend::LTX23, checkpoint;
                           tokenizer_model,
                           connector_checkpoint=backend.checkpoint,
                           device=:cpu, precision=:fp32,
                           tokenizer_library=sentencepiece_library())
    isempty(connector_checkpoint) &&
        throw(ArgumentError("LTX connector checkpoint path is empty"))
    tokenizer = SentencePieceTokenizer(tokenizer_model;
                                       library=tokenizer_library)
    try
        gemma = load_gemma3_text_encoder(String(checkpoint))
        connector = load_ltx23_text_connector(String(connector_checkpoint))
        transfer = array_transfer(device, precision)
        LTXTextConditioner(tokenizer,
            move_to_device(gemma, transfer),
            move_to_device(connector, transfer))
    catch
        close(tokenizer)
        rethrow()
    end
end

function encode_text(::LTX23, conditioner::LTXTextConditioner,
                     prompts::AbstractVector{<:AbstractString})
    tokenized = tokenize_gemma(conditioner.tokenizer, prompts;
                              max_length=conditioner.gemma.config.max_length)
    ltx23_text_conditioner_forward(conditioner, tokenized)
end

encode_text(backend::LTX23, conditioner::LTXTextConditioner,
            prompt::AbstractString) =
    encode_text(backend, conditioner, [prompt])

function encode_video(::LTX23, vae::LTXVideoVAEEncoder,
                      frames::AbstractArray)
    ndims(frames) in (4, 5) ||
        throw(DimensionMismatch("LTX frames must be (C,F,H,W) or (C,F,H,W,B)"))
    batched = ndims(frames) == 4 ?
        reshape(frames, size(frames)..., 1) : frames
    encoded = ltx23_vae_encoder_forward(vae, batched)
    ndims(frames) == 4 ? dropdims(encoded; dims=5) : encoded
end

function predict_velocity(::LTX23, model::LTXVideoTransformer,
                          latents, conditioning;
                          checkpoint_interval::Int=0)
    all(haskey(conditioning, key)
        for key in (:timesteps, :context, :positions)) ||
        throw(ArgumentError("LTX conditioning requires timesteps, context, and positions"))
    ltx_video_transformer_forward(model, latents,
        conditioning.timesteps, conditioning.context, conditioning.positions;
        checkpoint_interval=checkpoint_interval)
end

function lora_targets(::LTX23, model=nothing, selection=:attention)
    selection === :attention && return [
        r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)\.weight"]
    selection === :attention_and_ffn && return [
        r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)\.weight",
        r"transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)\.weight"]
    selection === :all_modal_attention && return [
        r"transformer_blocks\.\d+\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.(to_q|to_k|to_v|to_out\.0)\.weight"]
    selection isa AbstractVector && return selection
    throw(ArgumentError("unknown LTX-2.3 LoRA target preset $selection"))
end

export_mapping(::LTX23) = Dict(
    "prefix" => "diffusion_model.",
    "lora_A" => "lora_A.weight",
    "lora_B" => "lora_B.weight",
)
