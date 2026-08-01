@testset "Wan native CLIP vision encoder" begin
    config = WanCLIPVisionConfig(image_size=4, patch_size=2,
        hidden_size=8, intermediate_size=12, heads=2, layers=2,
        output_layer=1)
    model = WanCLIPVisionEncoder(config; rng=Xoshiro(201))
    images = randn(Xoshiro(202), Float32, 3, 4, 4, 2)
    encoded = wan_clip_vision_forward(model, images)
    @test size(encoded) == (8, 5, 2)
    @test all(isfinite, encoded)
    @test size(wan_clip_vision_forward(model, images[:, :, :, 1])) ==
        (8, 5)
    @test length(wan_clip_vision_specs(config)) == 40

    state = wan_clip_vision_state_dict(model)
    mktempdir() do directory
        path = joinpath(directory, "tiny-clip.safetensors")
        write_safetensors(path, state)
        loaded = load_wan_clip_vision(path, config)
        @test wan_clip_vision_forward(loaded, images) ≈ encoded
        @test wan_clip_vision_state_dict(loaded) == state
    end

    raw = reshape(Float32.(1:3 * 4 * 8), 3, 4, 8)
    processed = preprocess_wan_clip_image(raw; image_size=4)
    @test size(processed) == (3, 4, 4)
    @test all(isfinite, processed)
    features = encode_image(
        Wan21(variant=:i2v_14b_480p), model, zeros(Float32, 3, 6, 8))
    @test size(features) == (8, 5)

end
