function finite_difference!(f, x, index; epsilon=1f-3)
    indices = index isa Tuple ? index : (index,)
    old = x[indices...]
    x[indices...] = old + epsilon
    plus = f()
    x[indices...] = old - epsilon
    minus = f()
    x[indices...] = old
    (plus - minus) / (2epsilon)
end

@testset "mixed-precision activations preserve dtype" begin
    values = Float16[-2, -0.5, 0, 0.5, 2]
    activated = gelu_tanh(values)
    @test eltype(activated) === Float16
    @test Float32.(activated) ≈
        gelu_tanh(Float32.(values)) rtol=2f-3 atol=2f-4
end

@testset "RMSNorm" begin
    rng = Xoshiro(20)
    norm = RMSNorm(4)
    norm.weight .= randn(rng, Float32, 4)
    x = randn(rng, Float32, 4, 3, 2)
    dy = randn(rng, Float32, size(x))
    grads = rmsnorm_backward(norm, x, dy)
    objective() = sum(rmsnorm(norm, x) .* dy)
    @test grads.dx[2, 2, 1] ≈ finite_difference!(objective, x, (2, 2, 1)) rtol=3e-3
    @test grads.weight[3] ≈ finite_difference!(objective, norm.weight, 3) rtol=3e-3
    @test eltype(rmsnorm(norm, x)) == Float32
end

@testset "RoPE" begin
    rng = Xoshiro(21)
    rope = RotaryEmbedding(4)
    x = randn(rng, Float32, 4, 2, 3, 1)
    positions = Int32[0, 2, 7]
    y = apply_rope(rope, x, positions)
    @test sum(abs2, y; dims=1) ≈ sum(abs2, x; dims=1) rtol=2e-6
    @test rope_backward(rope, y, positions) ≈ x rtol=2e-6
end

@testset "modulation" begin
    rng = Xoshiro(22)
    x = randn(rng, Float32, 3, 4, 2)
    scale = randn(rng, Float32, 3, 2)
    shift = randn(rng, Float32, 3, 2)
    dy = randn(rng, Float32, size(x))
    grads = modulation_backward(x, scale, shift, dy)
    objective() = sum(modulation(x, scale, shift) .* dy)
    @test grads.dx[1, 2, 2] ≈ finite_difference!(objective, x, (1, 2, 2)) rtol=3e-3
    @test grads.scale[2, 1] ≈ finite_difference!(objective, scale, (2, 1)) rtol=3e-3
    @test grads.shift[3, 2] ≈ finite_difference!(objective, shift, (3, 2)) rtol=3e-3
end

@testset "reference attention" begin
    rng = Xoshiro(23)
    q = randn(rng, Float32, 4, 2, 3, 1)
    k = randn(rng, Float32, 4, 2, 3, 1)
    v = randn(rng, Float32, 4, 2, 3, 1)
    dy = randn(rng, Float32, 4, 2, 3, 1)
    result = reference_attention(q, k, v; causal=true)
    grads = attention_backward(q, k, v, dy, result.cache)
    objective() = sum(reference_attention(q, k, v; causal=true).output .* dy)
    @test grads.q[2, 1, 2, 1] ≈ finite_difference!(objective, q, (2, 1, 2, 1)) atol=2e-4 rtol=4e-3
    @test grads.k[3, 2, 2, 1] ≈ finite_difference!(objective, k, (3, 2, 2, 1)) atol=2e-4 rtol=4e-3
    @test grads.v[1, 1, 3, 1] ≈ finite_difference!(objective, v, (1, 1, 3, 1)) atol=2e-4 rtol=4e-3
    @test all(result.cache.probabilities[3, 1, :, :] .== 0)
    mask = Bool[1 1 1; 0 1 1; 0 0 1]
    @test reference_attention(q, k, v; mask=mask).output ≈ result.output
end

@testset "bounded-memory tiled attention" begin
    rng = Xoshiro(18)
    q = randn(rng, Float32, 4, 2, 7, 2)
    k = randn(rng, Float32, 4, 2, 9, 2)
    v = randn(rng, Float32, 4, 2, 9, 2)
    dy = randn(rng, Float32, 4, 2, 7, 2)
    mask = trues(9, 7)
    mask[8:9, 1:3] .= false
    reference = reference_attention(q, k, v; mask=mask)
    tiled = memory_efficient_attention(q, k, v; mask=mask,
        query_block=3, key_block=4)
    @test tiled.output ≈ reference.output rtol=2f-6 atol=2f-6
    @test size(tiled.cache.logsumexp) == (7, 2, 2)
    @test length(tiled.cache.logsumexp) < length(reference.cache.probabilities)
    expected_gradient = attention_backward(q, k, v, dy, reference.cache)
    actual_gradient =
        memory_efficient_attention_backward(q, k, v, dy, tiled.cache)
    @test actual_gradient.q ≈ expected_gradient.q rtol=3f-6 atol=3f-6
    @test actual_gradient.k ≈ expected_gradient.k rtol=3f-6 atol=3f-6
    @test actual_gradient.v ≈ expected_gradient.v rtol=3f-6 atol=3f-6

    self_q = randn(rng, Float32, 3, 1, 8, 1)
    self_v = randn(rng, Float32, 3, 1, 8, 1)
    causal_reference =
        reference_attention(self_q, self_q, self_v; causal=true)
    causal_tiled = memory_efficient_attention(
        self_q, self_q, self_v; causal=true, query_block=3, key_block=3)
    @test causal_tiled.output ≈ causal_reference.output rtol=2f-6 atol=2f-6
end
