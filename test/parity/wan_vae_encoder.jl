@testset "official Wan VAE encoder parity" begin
    fixture = joinpath(@__DIR__, "..", "fixtures",
        "wan21_vae_encoder_tiny.safetensors")
    provenance_path = joinpath(@__DIR__, "..", "fixtures",
        "wan21_vae_encoder_tiny.json")
    provenance = Reels.parse_json(read(provenance_path, String))
    @test provenance["upstream_commit"] ==
          "9737cba9c1c3c4d04b33fcad41c111989865d315"
    @test provenance["model_checkpoint"] == "Wan2.1_VAE.safetensors"
    @test provenance["seed"] == 119
    @test provenance["dtype"] == "float32"
    @test provenance["reels_layout"] == "C,T,H,W,B"

    source = open_tensor_source(fixture)
    config = WanVAEConfig(base_channels=2, latent_channels=16,
        channel_multipliers=[1, 2], residual_blocks=1,
        temporal_downsample=[true])
    vae = load_wan_vae(source, config; strict=false)
    model = vae.encoder
    input = load_state_tensor(source,
        TensorSpec("fixture.input", [3, 5, 4, 4, 1], ROW_MAJOR_SOURCE))
    expected = load_state_tensor(source,
        TensorSpec("fixture.output", [16, 3, 2, 2, 1], ROW_MAJOR_SOURCE))
    expected_decoded = load_state_tensor(source,
        TensorSpec("fixture.decoded", [3, 5, 4, 4, 1], ROW_MAJOR_SOURCE))
    expected_input_conv = load_state_tensor(source,
        TensorSpec("fixture.input_conv", [2, 5, 4, 4, 1], ROW_MAJOR_SOURCE))
    expected_first_residual = load_state_tensor(source,
        TensorSpec("fixture.first_residual", [2, 5, 4, 4, 1],
            ROW_MAJOR_SOURCE))
    actual_input_conv = model.input_conv(input)
    actual_first_residual =
        vae_residual_forward(first(model.downsample_layers), actual_input_conv)
    actual = wan_vae_encoder_forward(model, input)
    error = abs.(actual .- expected)
    @test collect(size(input)) == Int.(provenance["input_shape"])
    @test collect(size(expected)) == Int.(provenance["latent_shape"])
    @test size(actual) == size(expected)
    @test maximum(abs.(actual_input_conv .- expected_input_conv)) < 2f-6
    @test maximum(abs.(actual_first_residual .- expected_first_residual)) < 3f-6
    @test maximum(error) < provenance["max_encode_absolute_tolerance"]
    @test sum(error) / length(error) <
          provenance["mean_encode_absolute_tolerance"]
    decoded = wan_vae_decoder_forward(vae.decoder, expected)
    decode_error = abs.(decoded .- expected_decoded)
    @test collect(size(expected_decoded)) == Int.(provenance["decoded_shape"])
    @test size(decoded) == size(expected_decoded)
    @test maximum(decode_error) <
          provenance["max_decode_absolute_tolerance"]
    @test sum(decode_error) / length(decode_error) <
          provenance["mean_decode_absolute_tolerance"]
end
