@testset "LoRA linear" begin
    rng = Xoshiro(4)
    base = randn(rng, Float32, 3, 4)
    layer = LoRALinear(base; rank=2, alpha=2, rng=rng)
    x = randn(rng, Float32, 4, 5)
    @test layer(x) == base * x
    layer.B .= randn(rng, Float32, size(layer.B))
    @test layer(x) ≈ merged_weight(layer) * x
    set_adapter_enabled!(layer, false)
    @test layer(x) ≈ base * x
    set_adapter_enabled!(layer, true)
    dy = randn(rng, Float32, 3, 5)
    grads = lora_backward(layer, x, dy)
    eps = 1f-3
    old = layer.A[1]
    layer.A[1] = old + eps
    plus = sum(layer(x) .* dy)
    layer.A[1] = old - eps
    minus = sum(layer(x) .* dy)
    layer.A[1] = old
    @test grads.A[1] ≈ (plus - minus) / (2eps) rtol=2e-2

    layer.dropout = 0.5f0
    first = lora_forward(
        layer, x; training=true, dropout_seed=0x1234,
        dropout_stream=0x02)
    repeated = lora_forward(
        layer, x; training=true, dropout_seed=0x1234,
        dropout_stream=0x02)
    changed = lora_forward(
        layer, x; training=true, dropout_seed=0x1235,
        dropout_stream=0x02)
    @test first == repeated
    @test first != changed
    @test lora_forward(layer, x; training=false) ≈ merged_weight(layer) * x
end
