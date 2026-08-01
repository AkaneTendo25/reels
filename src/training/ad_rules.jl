import ChainRulesCore: AbstractZero, NoTangent, rrule, unthunk

function rrule(::typeof(_dropout_seed), seed::UInt64, stream::Integer)
    result = _dropout_seed(seed, stream)
    result, _ -> (NoTangent(), NoTangent(), NoTangent())
end

function rrule(::typeof(_dropout_mask_like), x::AbstractArray,
               probability::Float32, seed::UInt64, stream::UInt64)
    result = _dropout_mask_like(x, probability, seed, stream)
    result, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                  NoTangent())
end

# Treat the bit-safe BF16 conversion helpers as ordinary precision casts.
# Without explicit rules Zygote descends into GPUArrays' `reinterpret`
# bookkeeping and attempts to differentiate atomic reference counting.
function rrule(::typeof(float32_values),
               array::AbstractArray{BFloat16})
    result = float32_values(array)
    function float32_values_pullback(delta)
        delta = unthunk(delta)
        gradient = delta isa AbstractZero ? zero(array) :
            bfloat16_values(delta)
        (NoTangent(), gradient)
    end
    result, float32_values_pullback
end

function rrule(::typeof(bfloat16_values), array::AbstractArray)
    result = bfloat16_values(array)
    function bfloat16_values_pullback(delta)
        delta = unthunk(delta)
        gradient = delta isa AbstractZero ? zero(array) :
            cast_values(eltype(array), float32_values(delta))
        (NoTangent(), gradient)
    end
    result, bfloat16_values_pullback
end

function rrule(::typeof(cast_values), ::Type{BFloat16},
               array::AbstractArray)
    result = cast_values(BFloat16, array)
    function cast_bfloat16_pullback(delta)
        delta = unthunk(delta)
        gradient = delta isa AbstractZero ? zero(array) :
            cast_values(eltype(array), float32_values(delta))
        (NoTangent(), NoTangent(), gradient)
    end
    result, cast_bfloat16_pullback
end

# Frozen dense weights must still transmit activation gradients, but producing
# their parameter tangents wastes model-sized memory during LoRA training.
function rrule(::typeof(linear), layer::DenseLayer,
               x::AbstractArray)
    result = linear(layer, x)
    function linear_pullback(delta)
        delta = unthunk(delta)
        if delta isa AbstractZero
            return (NoTangent(), NoTangent(), zero(x))
        end
        delta_flat = reshape(delta, size(delta, 1), :)
        dx = reshape(_weight_transpose_mul(layer.weight, delta_flat), size(x))
        (NoTangent(), NoTangent(), dx)
    end
    result, linear_pullback
end

function rrule(::typeof(_weight_mul), weight::AbstractMatrix,
               input::AbstractMatrix)
    result = _weight_mul(weight, input)
    function frozen_weight_pullback(delta)
        delta = unthunk(delta)
        gradient = delta isa AbstractZero ? zero(input) :
            _weight_transpose_mul(weight, delta)
        (NoTangent(), NoTangent(), gradient)
    end
    result, frozen_weight_pullback
end

function rrule(::typeof(reference_attention), q::AbstractArray{T,4},
               k::AbstractArray{S,4}, v::AbstractArray{U,4};
               mask=nothing, causal::Bool=false) where {T,S,U}
    result = reference_attention(q, k, v; mask=mask, causal=causal)
    function reference_attention_pullback(delta)
        delta = unthunk(delta)
        output_delta = delta isa AbstractZero ? zero(result.output) :
            unthunk(getproperty(delta, :output))
        gradients = attention_backward(q, k, v, output_delta, result.cache)
        (NoTangent(), gradients.q, gradients.k, gradients.v)
    end
    result, reference_attention_pullback
end

function rrule(::typeof(memory_efficient_attention), q::AbstractArray{T,4},
               k::AbstractArray{S,4}, v::AbstractArray{U,4};
               mask=nothing, causal=false, query_block::Integer=128,
               key_block::Integer=128) where {T,S,U}
    result = memory_efficient_attention(q, k, v; mask=mask, causal=causal,
        query_block=query_block, key_block=key_block)
    function tiled_attention_pullback(delta)
        delta = unthunk(delta)
        output_delta = delta isa AbstractZero ? zero(result.output) :
            unthunk(getproperty(delta, :output))
        gradients = memory_efficient_attention_backward(
            q, k, v, output_delta, result.cache)
        (NoTangent(), gradients.q, gradients.k, gradients.v)
    end
    result, tiled_attention_pullback
end

function rrule(::typeof(wan_rope3d), x::AbstractArray{T,4}, grid_sizes;
               base=10_000f0, inverse=false) where T
    y = wan_rope3d(x, grid_sizes; base=base, inverse=inverse)
    function wan_rope3d_pullback(delta)
        dx = wan_rope3d(unthunk(delta), grid_sizes;
                        base=base, inverse=!inverse)
        (NoTangent(), dx, NoTangent())
    end
    y, wan_rope3d_pullback
end

function rrule(::typeof(sinusoidal_embedding), dimension::Integer, timesteps)
    y = sinusoidal_embedding(dimension, timesteps)
    y, _ -> (NoTangent(), NoTangent(), NoTangent())
end

function rrule(::typeof(ltx_timestep_embedding), timesteps, dimension::Integer)
    y = ltx_timestep_embedding(timesteps, dimension)
    y, _ -> (NoTangent(), NoTangent(), NoTangent())
end

function rrule(::typeof(ltx_rope_frequencies), config::LTX23Config,
               positions, like)
    y = ltx_rope_frequencies(config, positions, like)
    y, _ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent())
end

function rrule(::typeof(_constant_like), like, values)
    y = _constant_like(like, values)
    y, _ -> (NoTangent(), NoTangent(), NoTangent())
end
