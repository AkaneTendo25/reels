# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

"""
Configuration of the LTX-2.3 asymmetric transformer.

The defaults describe the official 22B audio-video model. `video_only=true`
keeps the video branch and its text cross-attention while omitting audio and
bidirectional audio-video blocks during construction/loading.
"""
struct LTX23Config
    variant::Symbol
    video_only::Bool
    video_heads::Int
    video_head_dim::Int
    video_channels::Int
    video_context_dim::Int
    audio_heads::Int
    audio_head_dim::Int
    audio_channels::Int
    audio_context_dim::Int
    layers::Int
    timestep_scale::Int
    rope_theta::Float32
    video_max_positions::NTuple{3,Int}
    audio_max_positions::NTuple{1,Int}
    cross_attention_adaln::Bool
    gated_attention::Bool
    epsilon::Float32
end

function LTX23Config(; variant=:av_22b_dev, video_only=false,
                     video_heads=32, video_head_dim=128,
                     video_channels=128, video_context_dim=4096,
                     audio_heads=32, audio_head_dim=64,
                     audio_channels=128, audio_context_dim=2048,
                     layers=48, timestep_scale=1000, rope_theta=10_000f0,
                     video_max_positions=(20, 2048, 2048),
                     audio_max_positions=(20,),
                     cross_attention_adaln=false, gated_attention=false,
                     epsilon=1f-6)
    video_heads > 0 && video_head_dim > 0 && audio_heads > 0 &&
        audio_head_dim > 0 || throw(ArgumentError("LTX attention dimensions must be positive"))
    layers > 0 || throw(ArgumentError("LTX layer count must be positive"))
    all(>(0), video_max_positions) ||
        throw(ArgumentError("LTX video position limits must be positive"))
    LTX23Config(Symbol(variant), Bool(video_only), Int(video_heads),
        Int(video_head_dim), Int(video_channels), Int(video_context_dim),
        Int(audio_heads), Int(audio_head_dim), Int(audio_channels),
        Int(audio_context_dim), Int(layers), Int(timestep_scale),
        Float32(rope_theta), Tuple(Int.(video_max_positions)),
        Tuple(Int.(audio_max_positions)), Bool(cross_attention_adaln),
        Bool(gated_attention), Float32(epsilon))
end

ltx23_video_dim(config::LTX23Config) =
    config.video_heads * config.video_head_dim
ltx23_audio_dim(config::LTX23Config) =
    config.audio_heads * config.audio_head_dim
ltx23_adaln_parameters(config::LTX23Config) =
    config.cross_attention_adaln ? 9 : 6

function _ltx_config_value(values, name, default)
    value = get(values, name, default)
    value === nothing ? default : value
end

function ltx23_config(values::AbstractDict; video_only=false)
    transformer = get(values, "transformer", values)
    max_video = Int.(_ltx_config_value(transformer,
        "positional_embedding_max_pos", [20, 2048, 2048]))
    max_audio = Int.(_ltx_config_value(transformer,
        "audio_positional_embedding_max_pos", [20]))
    length(max_video) == 3 ||
        throw(ArgumentError("LTX video positional_embedding_max_pos must have 3 entries"))
    length(max_audio) == 1 ||
        throw(ArgumentError("LTX audio positional_embedding_max_pos must have 1 entry"))
    LTX23Config(
        variant=Symbol(_ltx_config_value(values, "variant", "av_22b_dev")),
        video_only=video_only,
        video_heads=Int(_ltx_config_value(transformer, "num_attention_heads", 32)),
        video_head_dim=Int(_ltx_config_value(transformer, "attention_head_dim", 128)),
        video_channels=Int(_ltx_config_value(transformer, "in_channels", 128)),
        video_context_dim=Int(_ltx_config_value(transformer, "cross_attention_dim", 4096)),
        audio_heads=Int(_ltx_config_value(transformer, "audio_num_attention_heads", 32)),
        audio_head_dim=Int(_ltx_config_value(transformer, "audio_attention_head_dim", 64)),
        audio_channels=Int(_ltx_config_value(transformer, "audio_in_channels", 128)),
        audio_context_dim=Int(_ltx_config_value(transformer, "audio_cross_attention_dim", 2048)),
        layers=Int(_ltx_config_value(transformer, "num_layers", 48)),
        timestep_scale=Int(_ltx_config_value(transformer, "timestep_scale_multiplier", 1000)),
        rope_theta=Float32(_ltx_config_value(transformer, "positional_embedding_theta", 10_000)),
        video_max_positions=Tuple(max_video),
        audio_max_positions=Tuple(max_audio),
        cross_attention_adaln=Bool(_ltx_config_value(transformer,
            "cross_attention_adaln", false)),
        gated_attention=Bool(_ltx_config_value(transformer,
            "apply_gated_attention", false)))
end

function ltx23_config(source::AbstractString; video_only=false)
    values = if endswith(lowercase(source), ".safetensors")
        metadata = inspect_safetensors(source).metadata
        haskey(metadata, "config") ||
            throw(ArgumentError("LTX checkpoint has no `config` SafeTensors metadata"))
        parse_json(metadata["config"])
    else
        parse_json(read(source, String))
    end
    values isa AbstractDict ||
        throw(ArgumentError("LTX configuration root must be a JSON object"))
    ltx23_config(values; video_only=video_only)
end

ltx23_config(; video_only=false) = LTX23Config(video_only=video_only)
