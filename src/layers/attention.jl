"""Saved probabilities for the explicit reference-attention backward pass."""
struct AttentionCache{P}
    probabilities::P
    scale::Float32
end

struct TiledAttentionCache{L,O,M}
    logsumexp::L
    output::O
    mask::M
    scale::Float32
    query_block::Int
    key_block::Int
    causal::Bool
end

function _softmax_columns(scores)
    shifted = scores .- maximum(scores; dims=1)
    exps = exp.(shifted)
    exps ./ sum(exps; dims=1)
end

"""
    reference_attention(q, k, v; mask=nothing, causal=false)

Correct, allocation-heavy scaled dot-product attention. Inputs use the canonical
layout `(head_dim, heads, tokens, batch)`. A boolean mask has shape
`(key_tokens, query_tokens)` where `true` means visible.
"""
function reference_attention(q::AbstractArray{T,4}, k::AbstractArray{S,4},
                             v::AbstractArray{U,4}; mask=nothing,
                             causal::Bool=false) where {T,S,U}
    size(q, 1) == size(k, 1) == size(v, 1) ||
        throw(DimensionMismatch("q, k, and v head dimensions differ"))
    size(q, 2) == size(k, 2) == size(v, 2) ||
        throw(DimensionMismatch("q, k, and v head counts differ"))
    size(k, 3) == size(v, 3) ||
        throw(DimensionMismatch("k and v token counts differ"))
    size(q, 4) == size(k, 4) == size(v, 4) ||
        throw(DimensionMismatch("q, k, and v batch counts differ"))
    key_tokens, query_tokens = size(k, 3), size(q, 3)
    mask === nothing || size(mask) == (key_tokens, query_tokens) ||
        throw(DimensionMismatch("attention mask shape differs"))
    scale = inv(sqrt(Float32(size(q, 1))))
    out = similar(q, promote_type(T, S, U), size(v, 1), size(q, 2), query_tokens, size(q, 4))
    probabilities = similar(q, promote_type(T, S, U), key_tokens, query_tokens,
                            size(q, 2), size(q, 4))
    for batch in axes(q, 4), head in axes(q, 2)
        scores = (k[:, head, :, batch]' * q[:, head, :, batch]) .* scale
        if mask !== nothing
            scores = ifelse.(mask, scores, convert(eltype(scores), -Inf))
        end
        if causal
            for query in 1:query_tokens, key in 1:key_tokens
                key <= query || (scores[key, query] = convert(eltype(scores), -Inf))
            end
        end
        p = _softmax_columns(scores)
        probabilities[:, :, head, batch] .= p
        out[:, head, :, batch] .= v[:, head, :, batch] * p
    end
    (output=out, cache=AttentionCache(probabilities, scale))
end

function attention_backward(q::AbstractArray{T,4}, k::AbstractArray{S,4},
                            v::AbstractArray{U,4}, dy::AbstractArray{V,4},
                            cache::AttentionCache) where {T,S,U,V}
    size(dy) == (size(v, 1), size(q, 2), size(q, 3), size(q, 4)) ||
        throw(DimensionMismatch("attention output gradient shape differs"))
    dq, dk, dv = zero(q), zero(k), zero(v)
    for batch in axes(q, 4), head in axes(q, 2)
        p = cache.probabilities[:, :, head, batch]
        local_dy = dy[:, head, :, batch]
        local_v = v[:, head, :, batch]
        dprob = local_v' * local_dy
        dscores = p .* (dprob .- sum(dprob .* p; dims=1))
        dq[:, head, :, batch] .=
            (k[:, head, :, batch] * dscores) .* cache.scale
        dk[:, head, :, batch] .=
            (q[:, head, :, batch] * dscores') .* cache.scale
        dv[:, head, :, batch] .= local_dy * p'
    end
    (q=dq, k=dk, v=dv)
end

function _tile_visibility(mask, causal, keys, queries, like)
    mask === nothing && !causal && return nothing
    visible = mask === nothing ?
        trues(length(keys), length(queries)) :
        Array(view(mask, keys, queries))
    if causal
        visible .&= reshape(collect(keys), :, 1) .<=
            reshape(collect(queries), 1, :)
    end
    like isa CUDA.CuArray ? CUDA.CuArray(visible) : visible
end

"""
    memory_efficient_attention(q, k, v; query_block=128, key_block=128)

Exact scaled dot-product attention using online tiled softmax. Peak temporary
storage is `O(query_block * key_block)` rather than
`O(query_tokens * key_tokens)`. The cache stores only output and one
log-sum-exp value per query; backward recomputes score tiles.
"""
function memory_efficient_attention(q::AbstractArray{T,4},
                                    k::AbstractArray{S,4},
                                    v::AbstractArray{U,4};
                                    mask=nothing, causal=false,
                                    query_block::Integer=128,
                                    key_block::Integer=128) where {T,S,U}
    size(q, 1) == size(k, 1) == size(v, 1) ||
        throw(DimensionMismatch("q, k, and v head dimensions differ"))
    size(q, 2) == size(k, 2) == size(v, 2) ||
        throw(DimensionMismatch("q, k, and v head counts differ"))
    size(k, 3) == size(v, 3) ||
        throw(DimensionMismatch("k and v token counts differ"))
    size(q, 4) == size(k, 4) == size(v, 4) ||
        throw(DimensionMismatch("q, k, and v batch counts differ"))
    query_block > 0 && key_block > 0 ||
        throw(ArgumentError("attention tile sizes must be positive"))
    key_tokens, query_tokens = size(k, 3), size(q, 3)
    mask === nothing || size(mask) == (key_tokens, query_tokens) ||
        throw(DimensionMismatch("attention mask shape differs"))
    output_type = promote_type(T, S, U)
    output = similar(q, output_type, size(v, 1), size(q, 2),
        query_tokens, size(q, 4))
    logsumexp = similar(q, Float32, query_tokens, size(q, 2), size(q, 4))
    scale = inv(sqrt(Float32(size(q, 1))))
    for batch in axes(q, 4), head in axes(q, 2)
        for query_start in 1:Int(query_block):query_tokens
            queries = query_start:min(query_start + Int(query_block) - 1,
                query_tokens)
            local_q = float32_values(view(q, :, head, queries, batch))
            maximum_score = similar(local_q, Float32, 1, length(queries))
            fill!(maximum_score, -Inf32)
            normalizer = similar(maximum_score)
            fill!(normalizer, 0f0)
            accumulator = similar(local_q, Float32,
                size(v, 1), length(queries))
            fill!(accumulator, 0f0)
            for key_start in 1:Int(key_block):key_tokens
                keys = key_start:min(key_start + Int(key_block) - 1,
                    key_tokens)
                local_k = float32_values(view(k, :, head, keys, batch))
                scores = (transpose(local_k) * local_q) .* scale
                visible = _tile_visibility(mask, causal, keys, queries, scores)
                visible === nothing ||
                    (scores = ifelse.(visible, scores, -Inf32))
                block_maximum = maximum(scores; dims=1)
                next_maximum = max.(maximum_score, block_maximum)
                previous_scale = exp.(maximum_score .- next_maximum)
                probabilities = exp.(scores .- next_maximum)
                probabilities = ifelse.(isfinite.(scores), probabilities, 0f0)
                next_normalizer = previous_scale .* normalizer .+
                    sum(probabilities; dims=1)
                local_v = float32_values(view(v, :, head, keys, batch))
                accumulator .= accumulator .* previous_scale .+
                    local_v * probabilities
                maximum_score = next_maximum
                normalizer = next_normalizer
            end
            result = accumulator ./ normalizer
            view(output, :, head, queries, batch) .=
                cast_values(output_type, result)
            view(logsumexp, queries, head, batch) .=
                vec(maximum_score .+ log.(normalizer))
        end
    end
    (output=output, cache=TiledAttentionCache(logsumexp, output, mask,
        scale, Int(query_block), Int(key_block), causal))
end

function memory_efficient_attention_backward(q::AbstractArray{T,4},
                                             k::AbstractArray{S,4},
                                             v::AbstractArray{U,4},
                                             dy::AbstractArray{V,4},
                                             cache::TiledAttentionCache) where {T,S,U,V}
    size(dy) == size(cache.output) ||
        throw(DimensionMismatch("attention output gradient shape differs"))
    dq, dk, dv = zero(q), zero(k), zero(v)
    key_tokens, query_tokens = size(k, 3), size(q, 3)
    for batch in axes(q, 4), head in axes(q, 2)
        for query_start in 1:cache.query_block:query_tokens
            queries = query_start:min(query_start + cache.query_block - 1,
                query_tokens)
            local_q = float32_values(view(q, :, head, queries, batch))
            local_dy = float32_values(view(dy, :, head, queries, batch))
            local_output = float32_values(
                view(cache.output, :, head, queries, batch))
            correction = sum(local_dy .* local_output; dims=1)
            lse = reshape(view(cache.logsumexp, queries, head, batch), 1, :)
            query_gradient = similar(local_q)
            fill!(query_gradient, 0f0)
            for key_start in 1:cache.key_block:key_tokens
                keys = key_start:min(key_start + cache.key_block - 1,
                    key_tokens)
                local_k = float32_values(view(k, :, head, keys, batch))
                local_v = float32_values(view(v, :, head, keys, batch))
                scores = (transpose(local_k) * local_q) .* cache.scale
                visible = _tile_visibility(cache.mask, cache.causal,
                    keys, queries, scores)
                visible === nothing ||
                    (scores = ifelse.(visible, scores, -Inf32))
                probabilities = exp.(scores .- lse)
                probabilities =
                    ifelse.(isfinite.(scores), probabilities, 0f0)
                dprobability = transpose(local_v) * local_dy
                dscores = probabilities .* (dprobability .- correction)
                query_gradient .+=
                    (local_k * dscores) .* cache.scale
                view(dk, :, head, keys, batch) .+=
                    eltype(dk).((local_q * transpose(dscores)) .* cache.scale)
                view(dv, :, head, keys, batch) .+=
                    eltype(dv).(local_dy * transpose(probabilities))
            end
            view(dq, :, head, queries, batch) .= eltype(dq).(query_gradient)
        end
    end
    (q=dq, k=dk, v=dv)
end

# CUDA specialization using batched, tiled online softmax. The scalar host
# loops above provide the portable CPU implementation.
function memory_efficient_attention(
        q::CUDA.CuArray{T,4}, k::CUDA.CuArray{S,4},
        v::CUDA.CuArray{U,4}; mask=nothing, causal=false,
        query_block::Integer=512,
        key_block::Integer=512) where {T,S,U}
    size(q, 1) == size(k, 1) == size(v, 1) ||
        throw(DimensionMismatch("q, k, and v head dimensions differ"))
    size(q, 2) == size(k, 2) == size(v, 2) ||
        throw(DimensionMismatch("q, k, and v head counts differ"))
    size(k, 3) == size(v, 3) ||
        throw(DimensionMismatch("k and v token counts differ"))
    size(q, 4) == size(k, 4) == size(v, 4) ||
        throw(DimensionMismatch("q, k, and v batch counts differ"))
    query_block > 0 && key_block > 0 ||
        throw(ArgumentError("attention tile sizes must be positive"))
    key_tokens, query_tokens = size(k, 3), size(q, 3)
    mask === nothing || size(mask) == (key_tokens, query_tokens) ||
        throw(DimensionMismatch("attention mask shape differs"))
    output_type = promote_type(T, S, U)
    heads, batches = size(q, 2), size(q, 4)
    output = similar(q, output_type, size(v, 1), heads,
                     query_tokens, batches)
    logsumexp = similar(q, Float32, query_tokens, heads, batches)
    scale = inv(sqrt(Float32(size(q, 1))))

    for batch in axes(q, 4)
        for query_start in 1:Int(query_block):query_tokens
            queries = query_start:min(
                query_start + Int(query_block) - 1, query_tokens)
            # Internal batched-GEMM layout is feature × token × head.
            local_q = permutedims(float32_values(
                view(q, :, :, queries, batch)), (1, 3, 2))
            maximum_score = similar(
                local_q, Float32, 1, length(queries), heads)
            fill!(maximum_score, -Inf32)
            normalizer = similar(maximum_score)
            fill!(normalizer, 0f0)
            accumulator = similar(
                local_q, Float32, size(v, 1), length(queries), heads)
            fill!(accumulator, 0f0)

            for key_start in 1:Int(key_block):key_tokens
                keys = key_start:min(
                    key_start + Int(key_block) - 1, key_tokens)
                local_k = permutedims(float32_values(
                    view(k, :, :, keys, batch)), (1, 3, 2))
                scores = NNlib.batched_mul(
                    permutedims(local_k, (2, 1, 3)), local_q) .* scale
                visible = _tile_visibility(
                    mask, causal, keys, queries, scores)
                visible === nothing ||
                    (scores = ifelse.(
                        reshape(visible, length(keys), length(queries), 1),
                        scores, -Inf32))
                block_maximum = maximum(scores; dims=1)
                next_maximum = max.(maximum_score, block_maximum)
                previous_scale = exp.(maximum_score .- next_maximum)
                probabilities = exp.(scores .- next_maximum)
                probabilities =
                    ifelse.(isfinite.(scores), probabilities, 0f0)
                normalizer = previous_scale .* normalizer .+
                    sum(probabilities; dims=1)
                local_v = permutedims(float32_values(
                    view(v, :, :, keys, batch)), (1, 3, 2))
                accumulator .= accumulator .* previous_scale .+
                    NNlib.batched_mul(local_v, probabilities)
                maximum_score = next_maximum
            end

            result = accumulator ./ normalizer
            view(output, :, :, queries, batch) .= permutedims(
                cast_values(output_type, result), (1, 3, 2))
            view(logsumexp, queries, :, batch) .= dropdims(
                maximum_score .+ log.(normalizer); dims=1)
        end
    end
    (output=output, cache=TiledAttentionCache(
        logsumexp, output, mask, scale,
        Int(query_block), Int(key_block), causal))
end

function memory_efficient_attention_backward(
        q::CUDA.CuArray{T,4}, k::CUDA.CuArray{S,4},
        v::CUDA.CuArray{U,4}, dy::CUDA.CuArray{V,4},
        cache::TiledAttentionCache) where {T,S,U,V}
    size(dy) == size(cache.output) ||
        throw(DimensionMismatch("attention output gradient shape differs"))
    dq, dk, dv = zero(q), zero(k), zero(v)
    key_tokens, query_tokens = size(k, 3), size(q, 3)
    heads = size(q, 2)

    for batch in axes(q, 4)
        for query_start in 1:cache.query_block:query_tokens
            queries = query_start:min(
                query_start + cache.query_block - 1, query_tokens)
            local_q = permutedims(float32_values(
                view(q, :, :, queries, batch)), (1, 3, 2))
            local_dy = permutedims(float32_values(
                view(dy, :, :, queries, batch)), (1, 3, 2))
            local_output = permutedims(float32_values(
                view(cache.output, :, :, queries, batch)), (1, 3, 2))
            correction = sum(local_dy .* local_output; dims=1)
            lse = reshape(
                view(cache.logsumexp, queries, :, batch),
                1, length(queries), heads)
            query_gradient = similar(local_q)
            fill!(query_gradient, 0f0)

            for key_start in 1:cache.key_block:key_tokens
                keys = key_start:min(
                    key_start + cache.key_block - 1, key_tokens)
                local_k = permutedims(float32_values(
                    view(k, :, :, keys, batch)), (1, 3, 2))
                local_v = permutedims(float32_values(
                    view(v, :, :, keys, batch)), (1, 3, 2))
                scores = NNlib.batched_mul(
                    permutedims(local_k, (2, 1, 3)), local_q) .* cache.scale
                visible = _tile_visibility(
                    cache.mask, cache.causal, keys, queries, scores)
                visible === nothing ||
                    (scores = ifelse.(
                        reshape(visible, length(keys), length(queries), 1),
                        scores, -Inf32))
                probabilities = exp.(scores .- lse)
                probabilities =
                    ifelse.(isfinite.(scores), probabilities, 0f0)
                dprobability = NNlib.batched_mul(
                    permutedims(local_v, (2, 1, 3)), local_dy)
                dscores = probabilities .* (dprobability .- correction)
                query_gradient .+=
                    NNlib.batched_mul(local_k, dscores) .* cache.scale
                key_gradient = NNlib.batched_mul(
                    local_q, permutedims(dscores, (2, 1, 3))) .*
                    cache.scale
                value_gradient = NNlib.batched_mul(
                    local_dy, permutedims(probabilities, (2, 1, 3)))
                view(dk, :, :, keys, batch) .+= permutedims(
                    cast_values(eltype(dk), key_gradient), (1, 3, 2))
                view(dv, :, :, keys, batch) .+= permutedims(
                    cast_values(eltype(dv), value_gradient), (1, 3, 2))
            end
            view(dq, :, :, queries, batch) .= permutedims(
                cast_values(eltype(dq), query_gradient), (1, 3, 2))
        end
    end
    (q=dq, k=dk, v=dv)
end
