mutable struct LoRALinear{TW<:AbstractMatrix,TA<:AbstractMatrix,TB<:AbstractMatrix,TV}
    weight::TW
    bias::TV
    A::TA
    B::TB
    alpha::Float32
    dropout::Float32
    enabled::Bool
    train_bias::Bool
    base_bias::TV
end

function LoRALinear(weight::AbstractMatrix; rank::Integer, alpha::Real=rank,
                    dropout::Real=0, bias=nothing, train_bias::Bool=false,
                    rng=Random.default_rng())
    rank > 0 || throw(ArgumentError("rank must be positive"))
    isfinite(alpha) && alpha > 0 ||
        throw(ArgumentError("alpha must be finite and positive"))
    0 <= dropout < 1 || throw(ArgumentError("dropout must be in [0, 1)"))
    train_bias && bias === nothing &&
        throw(ArgumentError("cannot train a missing linear bias"))
    out, input = size(weight)
    A = randn(rng, Float32, rank, input) .* Float32(inv(sqrt(input)))
    B = zeros(Float32, out, rank)
    adapter_bias = train_bias ? copy(bias) : bias
    LoRALinear(weight, adapter_bias, A, B, Float32(alpha), Float32(dropout),
               true, train_bias, bias)
end

# Preserve the original positional constructor for downstream callers.
LoRALinear(weight, bias, A, B, alpha, dropout, enabled) =
    LoRALinear(weight, bias, A, B, alpha, dropout, enabled, false, bias)
LoRALinear(weight, bias, A, B, alpha, dropout, enabled, train_bias) =
    LoRALinear(weight, bias, A, B, alpha, dropout, enabled, train_bias,
               train_bias ? copy(bias) : bias)

lora_parameters(layer::LoRALinear) = layer.train_bias ?
    (A=layer.A, B=layer.B, bias=layer.bias) : (A=layer.A, B=layer.B)
set_adapter_enabled!(layer::LoRALinear, enabled::Bool) = (layer.enabled = enabled; layer)

function _dropout_seed(seed::UInt64, stream::Integer)
    seed ⊻ (UInt64(stream) * UInt64(0x9e3779b97f4a7c15))
end

function _dropout_mask_like(x::AbstractArray, probability::Float32,
                            seed::UInt64, stream::UInt64)
    rng = Xoshiro(_dropout_seed(seed, stream))
    host = Float32.(rand(rng, Float32, size(x)) .>= probability)
    if x isa CUDA.CuArray
        source = CUDA.CuArray(host)
        return cast_values(eltype(x), source)
    end
    cast_values(eltype(x), host)
end

function lora_forward(layer::LoRALinear, x::AbstractArray; training=false,
                      rng=Random.default_rng(), dropout_seed=nothing,
                      dropout_stream::Integer=0)
    size(x, 1) == size(layer.weight, 2) ||
        throw(DimensionMismatch("LoRA input and weight dimensions differ"))
    trailing = size(x)[2:end]
    x_flat = reshape(x, size(x, 1), :)
    y = _weight_mul(layer.weight, x_flat)
    active_bias = layer.train_bias && !layer.enabled ?
        layer.base_bias : layer.bias
    y = active_bias === nothing ? y :
        mixed_add(y, reshape(active_bias, :, 1))
    layer.enabled ||
        return reshape(y, size(layer.weight, 1), trailing...)
    z = x_flat
    if training && layer.dropout > 0
        mask = dropout_seed === nothing ?
            Float32.(rand(rng, Float32, size(x_flat)) .>= layer.dropout) :
            _dropout_mask_like(
                x_flat, layer.dropout, UInt64(dropout_seed),
                UInt64(dropout_stream))
        if !(mask isa typeof(x_flat)) && x_flat isa CUDA.CuArray
            mask = cast_values(eltype(x_flat), CUDA.CuArray(mask))
        else
            mask = cast_values(eltype(x_flat), mask)
        end
        kept = mixed_mul(x_flat, mask)
        z = cast_values(eltype(x_flat),
            float32_values(kept) ./ (1f0 - layer.dropout))
    end
    adapter = layer.B * (layer.A * z)
    if eltype(adapter) === BFloat16
        adapter32 = float32_values(adapter)
        scaled = Float32(layer.alpha / size(layer.A, 1)) .* adapter32
        adapter = bfloat16_values(scaled)
    else
        adapter = (layer.alpha / size(layer.A, 1)) .* adapter
    end
    reshape(mixed_add(y, adapter), size(layer.weight, 1), trailing...)
end

(layer::LoRALinear)(x::AbstractArray; kwargs...) = lora_forward(layer, x; kwargs...)

_projection(layer, x; training=false, dropout_seed=UInt64(0),
            dropout_stream=0) = layer(x)
_projection(layer::LoRALinear, x; training=false,
            dropout_seed::UInt64=UInt64(0), dropout_stream::Integer=0) =
    lora_forward(layer, x; training=training,
        dropout_seed=dropout_seed, dropout_stream=dropout_stream)

function lora_backward(layer::LoRALinear, x::AbstractArray, dy::AbstractArray)
    size(x)[2:end] == size(dy)[2:end] ||
        throw(DimensionMismatch("LoRA input and output trailing dimensions differ"))
    x_flat = reshape(x, size(x, 1), :)
    dy_flat = reshape(dy, size(dy, 1), :)
    scale = layer.alpha / size(layer.A, 1)
    ax = layer.A * x_flat
    dA = scale .* (layer.B' * dy_flat) * x_flat'
    dB = scale .* dy_flat * ax'
    dx = _weight_transpose_mul(layer.weight, dy_flat) +
        scale .* layer.A' * (layer.B' * dy_flat)
    (dx=reshape(dx, size(x)), A=dA, B=dB)
end

merged_weight(layer::LoRALinear) =
    Array(layer.weight) +
    (layer.alpha / size(layer.A, 1)) .* (Array(layer.B) * Array(layer.A))
