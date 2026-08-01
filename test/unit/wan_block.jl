@testset "Wan 2.1 transformer block" begin
    rng = Xoshiro(31)
    block = WanTransformerBlock(12, 24, 3; rng=rng)
    x = randn(rng, Float32, 12, 4, 2)
    time = randn(rng, Float32, 12, 6, 2)
    context = randn(rng, Float32, 12, 5, 2)
    grids = [(1, 2, 2), (1, 2, 2)]
    y = wan_block_forward(block, x, time, context, grids)
    @test size(y) == size(x)
    @test eltype(y) == Float32
    @test all(isfinite, y)

    heads = randn(rng, Float32, 4, 3, 4, 2)
    rotated = wan_rope3d(heads, grids)
    @test sum(abs2, rotated; dims=1) ≈ sum(abs2, heads; dims=1) rtol=2e-6
    @test wan_rope3d(rotated, grids; inverse=true) ≈ heads rtol=2e-6

    block.self_attention.o.weight .= 0
    block.self_attention.o.bias .= 0
    block.cross_attention.o.weight .= 0
    block.cross_attention.o.bias .= 0
    block.ffn_out.weight .= 0
    block.ffn_out.bias .= 0
    @test wan_block_forward(block, x, time, context, grids) == x
end
