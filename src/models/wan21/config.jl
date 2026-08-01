struct Wan21Config
    variant::Symbol
    model_type::Symbol
    patch_size::NTuple{3,Int}
    text_length::Int
    input_channels::Int
    hidden_size::Int
    ffn_size::Int
    frequency_size::Int
    text_size::Int
    output_channels::Int
    heads::Int
    layers::Int
    window_size::NTuple{2,Int}
    qk_norm::Bool
    cross_attention_norm::Bool
    epsilon::Float32
end

function Wan21Config(; variant=:t2v_1_3b, model_type=:t2v,
                     patch_size=(1, 2, 2), text_length=512,
                     input_channels=16, hidden_size=1536, ffn_size=8960,
                     frequency_size=256, text_size=4096, output_channels=16,
                     heads=12, layers=30, window_size=(-1, -1),
                     qk_norm=true, cross_attention_norm=true, epsilon=1f-6)
    hidden_size % heads == 0 ||
        throw(ArgumentError("hidden_size must be divisible by heads"))
    iseven(hidden_size ÷ heads) ||
        throw(ArgumentError("Wan head dimension must be even"))
    Wan21Config(variant, model_type, patch_size, text_length, input_channels,
        hidden_size, ffn_size, frequency_size, text_size, output_channels,
        heads, layers, window_size, qk_norm, cross_attention_norm,
        Float32(epsilon))
end

function wan21_config(variant::Symbol)
    variant === :t2v_1_3b && return Wan21Config()
    variant === :t2v_14b && return Wan21Config(variant=:t2v_14b,
        hidden_size=5120, ffn_size=13824, heads=40, layers=40)
    variant in (:i2v_14b_480p, :i2v_14b_720p) &&
        return Wan21Config(variant=variant, model_type=:i2v,
            input_channels=36, hidden_size=5120, ffn_size=13824,
            heads=40, layers=40)
    throw(ArgumentError("unsupported Wan 2.1 variant $variant"))
end
