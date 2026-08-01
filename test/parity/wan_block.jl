@testset "official Wan 2.1 block parity" begin
    fixture = normpath(joinpath(@__DIR__, "..", "fixtures",
                                "wan21_block_tiny.safetensors"))
    provenance = Reels.parse_json(read(replace(fixture, ".safetensors" => ".json"),
                                       String))
    header = inspect_safetensors(fixture)
    @test header.metadata["upstream_commit"] == provenance["upstream_commit"]
    @test header.metadata["seed"] == provenance["seed"]
    @test header.metadata["dtype"] == provenance["dtype"]
    @test provenance["model_checkpoint"] ==
          "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    @test header.tensors["fixture.x"].shape ==
          Int.(provenance["input_shape"])
    @test header.tensors["fixture.output"].shape ==
          Int.(provenance["output_shape"])

    config = Wan21Config(hidden_size=24, ffn_size=32, frequency_size=8,
        text_size=24, heads=3, layers=1)
    block = load_wan_block(fixture, "blocks.", config)
    x = load_safetensor(fixture, "fixture.x")
    time_modulation = load_safetensor(fixture, "fixture.time_modulation")
    context = load_safetensor(fixture, "fixture.context")
    grid_tensor = load_safetensor(fixture, "fixture.grid_sizes")
    grids = [Tuple(Int.(grid_tensor[:, batch])) for batch in axes(grid_tensor, 2)]
    expected = load_safetensor(fixture, "fixture.output")

    actual = wan_block_forward(block, x, time_modulation, context, grids)
    absolute_error = abs.(actual .- expected)
    @test maximum(absolute_error) < provenance["max_absolute_tolerance"]
    @test sum(absolute_error) / length(absolute_error) <
          provenance["mean_absolute_tolerance"]
    @test actual ≈ expected atol=provenance["max_absolute_tolerance"] rtol=1f-5
end
