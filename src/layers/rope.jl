"""Rotary positional embedding configuration."""
struct RotaryEmbedding
    dimension::Int
    base::Float32
end

function RotaryEmbedding(dimension::Integer; base=10_000f0)
    iseven(dimension) || throw(ArgumentError("RoPE dimension must be even"))
    dimension > 0 || throw(ArgumentError("RoPE dimension must be positive"))
    RotaryEmbedding(Int(dimension), Float32(base))
end

function _rope!(y, rope::RotaryEmbedding, x, positions, inverse::Bool)
    size(x, 1) >= rope.dimension ||
        throw(DimensionMismatch("input feature dimension is smaller than RoPE dimension"))
    ndims(x) == 4 || throw(DimensionMismatch("RoPE expects (head_dim, heads, tokens, batch)"))
    length(positions) == size(x, 3) ||
        throw(DimensionMismatch("position count and token count differ"))
    copyto!(y, x)
    sign = inverse ? -1f0 : 1f0
    half = rope.dimension ÷ 2
    for batch in axes(x, 4), token in axes(x, 3), head in axes(x, 2), pair in 1:half
        i, j = 2pair - 1, 2pair
        theta = sign * Float32(positions[token]) /
            rope.base^(Float32(pair - 1) / Float32(half))
        c, s = cos(theta), sin(theta)
        a, b = x[i, head, token, batch], x[j, head, token, batch]
        y[i, head, token, batch] = c * a - s * b
        y[j, head, token, batch] = s * a + c * b
    end
    y
end

apply_rope(rope::RotaryEmbedding, x::AbstractArray, positions=0:size(x, 3)-1) =
    _rope!(similar(x), rope, x, positions, false)

rope_backward(rope::RotaryEmbedding, dy::AbstractArray, positions=0:size(dy, 3)-1) =
    _rope!(similar(dy), rope, dy, positions, true)
