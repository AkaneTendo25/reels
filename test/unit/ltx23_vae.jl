@testset "LTX-2.3 native video VAE encoder" begin
    parsed = ltx23_vae_config(Dict("vae" => Dict(
        "in_channels" => 3,
        "latent_channels" => 128,
        "patch_size" => 4,
        "norm_layer" => "pixel_norm",
        "latent_log_var" => "uniform",
        "spatial_padding_mode" => "zeros",
        "encoder_blocks" => [
            ["res_x", Dict("num_layers" => 4)],
            ["compress_space_res", Dict("multiplier" => 2)],
            ["res_x", Dict("num_layers" => 6)],
            ["compress_time_res", Dict("multiplier" => 2)],
            ["res_x", Dict("num_layers" => 4)],
            ["compress_all_res", Dict("multiplier" => 2)],
            ["res_x", Dict("num_layers" => 2)],
            ["compress_all_res", Dict("multiplier" => 1)],
            ["res_x", Dict("num_layers" => 2)],
        ])))
    @test parsed.patch_size == 4
    @test length(parsed.blocks) == 9
    @test sum(block.num_layers for block in parsed.blocks) == 18
    @test length(ltx23_vae_encoder_specs(parsed)) == 86
    @test first(ltx23_vae_encoder_specs(parsed)).source_key ==
        "vae.encoder.conv_in.conv.weight"
    @test last(ltx23_vae_encoder_specs(parsed)).source_key ==
        "vae.per_channel_statistics.std-of-means"
    parsed_specs = Dict(spec.source_key => spec.source_shape
                        for spec in ltx23_vae_encoder_specs(parsed))
    @test parsed_specs["vae.encoder.conv_in.conv.weight"] ==
        [128, 48, 3, 3, 3]
    @test parsed_specs["vae.encoder.down_blocks.1.conv.conv.weight"] ==
        [64, 128, 3, 3, 3]
    @test parsed_specs["vae.encoder.down_blocks.3.conv.conv.weight"] ==
        [256, 256, 3, 3, 3]
    @test parsed_specs["vae.encoder.down_blocks.5.conv.conv.weight"] ==
        [128, 512, 3, 3, 3]
    @test parsed_specs["vae.encoder.down_blocks.7.conv.conv.weight"] ==
        [128, 1024, 3, 3, 3]
    @test parsed_specs["vae.encoder.conv_out.conv.weight"] ==
        [129, 1024, 3, 3, 3]

    image = reshape(Float32.(1:4), 1, 1, 2, 2, 1)
    @test vec(ltx23_vae_patchify(image, 2)) == Float32[1, 2, 3, 4]

    causal = VAEConv3D(ones(Float32, 1, 1, 3, 1, 1),
        zeros(Float32, 1); padding=(1, 0, 0))
    temporal = reshape(Float32[1, 2, 3], 1, 3, 1, 1, 1)
    @test vec(ltx23_causal_conv3d(causal, temporal)) ==
        Float32[3, 4, 6]

    config = LTXVideoVAEConfig(input_channels=3, latent_channels=4,
        patch_size=2, blocks=LTXVAEBlockConfig[
            LTXVAEBlockConfig(:res_x, 1, 1),
            LTXVAEBlockConfig(:compress_space_res, 0, 2),
            LTXVAEBlockConfig(:res_x, 1, 1),
            LTXVAEBlockConfig(:compress_time_res, 0, 2),
            LTXVAEBlockConfig(:compress_all_res, 0, 2),
        ])
    @test length(ltx23_vae_encoder_specs(config)) == 20
    model = LTXVideoVAEEncoder(config; rng=Xoshiro(230))
    video = randn(Xoshiro(231), Float32, 3, 9, 8, 8, 1)
    encoded = ltx23_vae_encoder_forward(model, video)
    @test size(encoded) == (4, 3, 1, 1, 1)
    @test all(isfinite, encoded)
    @test encode_video(LTX23(video_only=true), model,
        dropdims(video; dims=5)) ≈ dropdims(encoded; dims=5)

    cropped = ltx23_vae_encoder_forward(model,
        cat(video, video[:, 1:2, :, :, :]; dims=2))
    @test cropped ≈ encoded

    zero_residual = LTXVAEResidualBlock(
        VAEConv3D(zeros(Float32, 2, 2, 3, 3, 3), zeros(Float32, 2);
                  padding=(1, 1, 1)),
        VAEConv3D(zeros(Float32, 2, 2, 3, 3, 3), zeros(Float32, 2);
                  padding=(1, 1, 1)))
    residual_input = randn(Xoshiro(232), Float32, 2, 3, 2, 2, 1)
    @test ltx23_vae_residual_forward(zero_residual, residual_input) ==
        residual_input

    mktempdir() do directory
        specs = ltx23_vae_encoder_specs(config)
        state = Dict{String,AbstractArray}(
            spec.source_key => zeros(Float32, spec.source_shape...)
            for spec in specs)
        state["vae.per_channel_statistics.std-of-means"] .= 1f0
        path = joinpath(directory, "tiny-ltx-vae.safetensors")
        write_safetensors(path, state)
        source = open_tensor_source(path)
        @test isempty(audit_state_dict(source, specs))
        loaded = load_ltx23_vae_encoder(source, config; strict=true)
        @test size(loaded.conv_out.weight) == (5, 32, 3, 3, 3)
        @test ltx23_vae_encoder_forward(loaded, video) ==
            zeros(Float32, 4, 3, 1, 1, 1)
    end
end
