"""RMS normalization over the first (feature) dimension."""
struct RMSNorm{T<:AbstractVector}
    weight::T
    epsilon::Float32
end

RMSNorm(features::Integer; epsilon=1f-6) =
    RMSNorm(ones(Float32, features), Float32(epsilon))

function rmsnorm(norm::RMSNorm, x::AbstractArray)
    size(x, 1) == length(norm.weight) ||
        throw(DimensionMismatch("RMSNorm weight and feature dimensions differ"))
    x32 = float32_values(x)
    inv_rms = inv.(sqrt.(sum(abs2, x32; dims=1) ./
                          Float32(size(x, 1)) .+ norm.epsilon))
    normalized = cast_values(eltype(x), x32 .* inv_rms)
    weight = reshape(norm.weight, :, ntuple(_ -> 1, ndims(x) - 1)...)
    if eltype(x) === BFloat16
        weight32 = float32_values(weight)
        normalized32 = float32_values(normalized)
        return bfloat16_values(weight32 .* normalized32)
    end
    weight .* normalized
end

(norm::RMSNorm)(x::AbstractArray) = rmsnorm(norm, x)

function rmsnorm_backward(norm::RMSNorm, x::AbstractArray, dy::AbstractArray)
    size(x) == size(dy) || throw(DimensionMismatch("x and dy shapes differ"))
    n = size(x, 1)
    T = promote_type(eltype(x), eltype(dy), eltype(norm.weight))
    w = reshape(norm.weight, :, ntuple(_ -> 1, ndims(x) - 1)...)
    inv_rms = inv.(sqrt.(sum(abs2, x; dims=1) ./ T(n) .+ T(norm.epsilon)))
    weighted_dy = dy .* w
    dot_term = sum(weighted_dy .* x; dims=1)
    dx = weighted_dy .* inv_rms .- x .* (inv_rms .^ 3) .* dot_term ./ T(n)
    reduce_dims = Tuple(2:ndims(x))
    dweight = vec(sum(dy .* x .* inv_rms; dims=reduce_dims))
    (dx=dx, weight=dweight)
end
