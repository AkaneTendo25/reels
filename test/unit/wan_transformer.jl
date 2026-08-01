@testset "Wan 2.1 tiny full transformer" begin
    @test Reels._wan_model_timesteps(Float32[0, 0.4, 1], false) ==
          Float32[0, 400, 1000]
    @test Reels._wan_model_timesteps(Float32[0, 0.4, 1], true) ==
          Float32[1, 401, 1001]
    silu_input = reshape(Float32[-2, 0, 2], :, 1)
    @test Reels._wan_silu(silu_input) ≈
          silu_input ./ (1f0 .+ exp.(-silu_input))

    rng = Xoshiro(41)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=5,
        input_channels=2, hidden_size=12, ffn_size=24, frequency_size=8,
        text_size=10, output_channels=2, heads=3, layers=2)
    model = WanTransformer(config; rng=rng)
    video = randn(rng, Float32, 2, 2, 4, 4, 2)
    text = randn(rng, Float32, 10, 5, 2)
    output = wan_transformer_forward(model, video, Float32[0.2, 0.7], text)
    @test size(output) == size(video)
    @test eltype(output) == Float32
    @test all(isfinite, output)
    @test all(iszero, output) # official initialization zeros the output head

    model.head.weight .= randn(rng, Float32, size(model.head.weight))
    changed = wan_transformer_forward(model, video, Float32[0.2, 0.7], text)
    @test any(!iszero, changed)
    @test changed != wan_transformer_forward(model, video, Float32[0.3, 0.7], text)
end

@testset "Wan 2.1 tiny I2V transformer" begin
    rng = MersenneTwister(707)
    config = Wan21Config(
        variant=:i2v_test, model_type=:i2v,
        patch_size=(1, 2, 2), text_length=2,
        input_channels=36, hidden_size=8, ffn_size=16,
        frequency_size=4, text_size=6, output_channels=16,
        heads=1, layers=1)
    model = WanTransformer(config; rng=rng)
    model.head.weight .=
        randn(rng, Float32, size(model.head.weight))
    noisy = randn(rng, Float32, 16, 1, 2, 4, 1)
    conditioning = randn(rng, Float32, 20, 1, 2, 4, 1)
    text = randn(rng, Float32, 6, 2, 1)
    image = randn(rng, Float32, 1280, 3, 1)
    timesteps = Float32[0.3]

    output = wan_transformer_forward(
        model, noisy, timesteps, text;
        conditioning_video=conditioning,
        image_features=image)
    changed = wan_transformer_forward(
        model, noisy, timesteps, text;
        conditioning_video=conditioning,
        image_features=image .+ 0.25f0)
    @test size(output) == size(noisy)
    @test all(isfinite, output)
    @test changed != output
    @test_throws ArgumentError wan_transformer_forward(
        model, noisy, timesteps, text)

    state = wan_transformer_state_dict(model)
    @test length(state) == length(wan21_transformer_specs(config))
    mktempdir() do directory
        checkpoint = joinpath(directory, "i2v.safetensors")
        write_safetensors(checkpoint, state)
        restored = load_wan_transformer(checkpoint, config)
        actual = wan_transformer_forward(
            restored, noisy, timesteps, text;
            conditioning_video=conditioning,
            image_features=image)
        @test actual ≈ output rtol=1f-5 atol=1f-5
    end
end

@testset "streaming sharded Wan transformer load" begin
    rng = Xoshiro(42)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=5,
        input_channels=2, hidden_size=12, ffn_size=24, frequency_size=8,
        text_size=10, output_channels=2, heads=3, layers=2)
    original = WanTransformer(config; rng=rng)
    original.head.weight .= randn(rng, Float32, size(original.head.weight))
    original.head.bias .= randn(rng, Float32, size(original.head.bias))
    state = wan_transformer_state_dict(original)
    keys_sorted = sort!(collect(keys(state)))
    midpoint = length(keys_sorted) ÷ 2
    shard1 = Dict(key => state[key] for key in keys_sorted[1:midpoint])
    shard2 = Dict(key => state[key] for key in keys_sorted[midpoint+1:end])
    mktempdir() do dir
        index = joinpath(dir, "diffusion_pytorch_model.safetensors.index.json")
        write_sharded_safetensors(index, Dict(
            "diffusion_pytorch_model-00001-of-00002.safetensors" => shard1,
            "diffusion_pytorch_model-00002-of-00002.safetensors" => shard2))
        source = open_tensor_source(index)
        @test isempty(audit_state_dict(source, wan21_transformer_specs(config)))
        loaded = load_wan_transformer(source, config)
        video = randn(rng, Float32, 2, 2, 4, 4, 1)
        text = randn(rng, Float32, 10, 5, 1)
        timestep = Float32[0.4]
        expected = wan_transformer_forward(original, video, timestep, text)
        actual = wan_transformer_forward(loaded, video, timestep, text)
        @test actual == expected
        @test wan_transformer_state_dict(loaded) == state
    end
end
