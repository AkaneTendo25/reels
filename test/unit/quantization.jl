@testset "frozen Int8 weights and CPU offload storage" begin
    rng = Xoshiro(950)
    weight = randn(rng, Float32, 7, 5)
    input = randn(rng, Float32, 5, 4)
    quantized = quantize_frozen_matrix(weight, Float32)
    @test quantized isa QuantizedMatrix
    @test size(quantized) == size(weight)
    @test Array(quantized) ≈ weight rtol=1f-2 atol=1f-2

    dense = DenseLayer(quantized, zeros(Float32, 7))
    @test linear(dense, input) ≈ weight * input rtol=2f-2 atol=2f-2

    offloaded = CPUOffloadedMatrix{Float32,Matrix{Float32}}(weight)
    @test Reels._weight_mul(offloaded, input) == weight * input
    @test Reels._weight_transpose_mul(offloaded, randn(rng, Float32, 7, 4)) isa
        Matrix{Float32}

    layer = LoRALinear(quantized; rank=2, alpha=2, rng=rng)
    layer.B .= randn(rng, Float32, size(layer.B))
    reference = LoRALinear(weight, nothing, copy(layer.A), copy(layer.B),
                           layer.alpha, layer.dropout, layer.enabled)
    @test lora_forward(layer, input) ≈
        lora_forward(reference, input) rtol=2f-2 atol=2f-2

    result = Reels.Zygote.withgradient(
            Reels.Zygote.Params([layer.A, layer.B])) do
        sum(abs2, lora_forward(layer, input))
    end
    @test result.grad[layer.A] !== nothing
    @test result.grad[layer.B] !== nothing
    @test all(isfinite, result.grad[layer.A])
    @test all(isfinite, result.grad[layer.B])
end

@testset "low-VRAM model transfer quantizes only frozen dense weights" begin
    rng = Xoshiro(951)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
        text_size=5, output_channels=1, heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    adapted = inject_wan_lora(base;
        targets=["blocks.0.self_attn.q.weight"],
        rank=1, alpha=1, rng=rng)
    transfer = frozen_weight_transfer(
        :cpu, :fp32; quantization=:int8)
    moved = move_to_device(adapted, transfer)
    @test moved.blocks[1].self_attention.q.weight isa QuantizedMatrix
    @test moved.blocks[1].self_attention.q.A isa Matrix{Float32}
    @test moved.blocks[1].self_attention.k.weight isa QuantizedMatrix
    rematerialized = move_to_device(moved, transfer)
    @test rematerialized.blocks[1].self_attention.q.weight isa QuantizedMatrix
    @test rematerialized.blocks[1].self_attention.q.weight.values ==
        moved.blocks[1].self_attention.q.weight.values
    @test_throws ArgumentError frozen_weight_transfer(
        :cpu, :fp32; cpu_offload=true)
    @test_throws ArgumentError frozen_weight_transfer(
        :cuda, :bf16; quantization=:int8)
end
