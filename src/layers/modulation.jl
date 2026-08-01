"""Apply AdaLN-style scale and shift to `(features, tokens, batch)` activations."""
function modulation(x::AbstractArray{T,3}, scale::AbstractMatrix,
                    shift::AbstractMatrix) where T
    size(scale) == (size(x, 1), size(x, 3)) ||
        throw(DimensionMismatch("scale must have shape (features, batch)"))
    size(shift) == size(scale) ||
        throw(DimensionMismatch("shift and scale shapes differ"))
    mixed_affine(x,
        reshape(scale, size(scale, 1), 1, size(scale, 2)),
        reshape(shift, size(shift, 1), 1, size(shift, 2)))
end

function modulation_backward(x::AbstractArray{T,3}, scale::AbstractMatrix,
                             shift::AbstractMatrix, dy::AbstractArray{S,3}) where {T,S}
    size(x) == size(dy) || throw(DimensionMismatch("x and dy shapes differ"))
    size(scale) == size(shift) == (size(x, 1), size(x, 3)) ||
        throw(DimensionMismatch("modulation parameter shapes differ"))
    expanded_scale = reshape(1 .+ scale, size(scale, 1), 1, size(scale, 2))
    (dx=dy .* expanded_scale,
     scale=dropdims(sum(dy .* x; dims=2); dims=2),
     shift=dropdims(sum(dy; dims=2); dims=2))
end
