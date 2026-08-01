struct DenseLayer{W<:AbstractMatrix,B}
    weight::W
    bias::B
end

_mixed_precision_type(array::AbstractArray) = eltype(array) === BFloat16
function mixed_add(left::AbstractArray, right::AbstractArray)
    _mixed_precision_type(left) || return left .+ right
    left32 = float32_values(left)
    right32 = float32_values(right)
    summed = left32 .+ right32
    bfloat16_values(summed)
end
function mixed_sub(left::AbstractArray, right::AbstractArray)
    _mixed_precision_type(left) || return left .- right
    difference = float32_values(left) .- float32_values(right)
    bfloat16_values(difference)
end
function mixed_mul(left::AbstractArray, right::AbstractArray)
    _mixed_precision_type(left) || return left .* right
    left32 = float32_values(left)
    right32 = float32_values(right)
    product = left32 .* right32
    bfloat16_values(product)
end
function mixed_affine(x::AbstractArray, scale::AbstractArray,
                      shift::AbstractArray)
    _mixed_precision_type(x) ||
        return x .* (1f0 .+ scale) .+ shift
    x32 = float32_values(x)
    scale32 = float32_values(scale)
    shift32 = float32_values(shift)
    scaled = x32 .* (1f0 .+ scale32)
    shifted = scaled .+ shift32
    bfloat16_values(shifted)
end

function linear(layer::DenseLayer, x::AbstractArray)
    size(x, 1) == size(layer.weight, 2) ||
        throw(DimensionMismatch("linear input and weight dimensions differ"))
    trailing = size(x)[2:end]
    y = _weight_mul(layer.weight, reshape(x, size(x, 1), :))
    y = layer.bias === nothing ? y :
        mixed_add(y, reshape(layer.bias, :, 1))
    reshape(y, size(layer.weight, 1), trailing...)
end
(layer::DenseLayer)(x::AbstractArray) = linear(layer, x)

function layernorm(x::AbstractArray; epsilon=1f-6, weight=nothing, bias=nothing)
    x32 = float32_values(x)
    mean_x = sum(x32; dims=1) ./ Float32(size(x, 1))
    centered = x32 .- mean_x
    variance = sum(abs2, centered; dims=1) ./ Float32(size(x, 1))
    y = centered ./ sqrt.(variance .+ Float32(epsilon))
    shape = (:, ntuple(_ -> 1, ndims(x) - 1)...)
    y = weight === nothing ? y : y .* reshape(float32_values(weight), shape...)
    y = bias === nothing ? y : y .+ reshape(float32_values(bias), shape...)
    cast_values(eltype(x), y)
end

function gelu_tanh(x::AbstractArray)
    x32 = float32_values(x)
    cubic = x32 .^ 3
    inner = x32 .+ 0.044715f0 .* cubic
    activated = tanh.(sqrt(Float32(2 / pi)) .* inner)
    y = 0.5f0 .* x32 .* (1f0 .+ activated)
    cast_values(eltype(x), y)
end
