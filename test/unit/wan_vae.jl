@testset "Wan VAE primitives" begin
    temporal_weight = ones(Float32, 1, 1, 3, 1, 1)
    temporal = VAEConv3D(temporal_weight, zeros(Float32, 1);
        padding=(1, 0, 0))
    input = reshape(Float32[1, 2, 3], 1, 3, 1, 1, 1)
    output = vae_conv3d(temporal, input)
    @test size(output) == size(input)
    @test vec(output) == Float32[1, 3, 6]

    temporal_cache = Reels.VAEDecodeFeatureCache()
    streamed = Any[]
    for frame in axes(input, 2)
        temporal_cache.index = 0
        push!(streamed, Reels._vae_cached_conv3d(
            temporal, view(input, :, frame:frame, :, :, :),
            temporal_cache))
    end
    @test cat(streamed...; dims=2) == output

    spatial_identity = VAEConv3D(
        ones(Float32, 1, 1, 1, 1, 1), zeros(Float32, 1))
    temporal_expand = VAEConv3D(
        ones(Float32, 2, 1, 3, 1, 1), zeros(Float32, 2);
        padding=(1, 0, 0))
    upsample = VAEUpsample(spatial_identity, temporal_expand)
    expected_upsample = vae_upsample_forward(upsample, input)
    upsample_cache = Reels.VAEDecodeFeatureCache()
    streamed_upsample = Any[]
    for frame in axes(input, 2)
        upsample_cache.index = 0
        push!(streamed_upsample, Reels.vae_upsample_forward(
            upsample, view(input, :, frame:frame, :, :, :),
            upsample_cache))
    end
    @test cat(streamed_upsample...; dims=2) == expected_upsample

    spatial_weight = ones(Float32, 1, 1, 1, 3, 3)
    spatial = VAEConv3D(spatial_weight, zeros(Float32, 1);
        stride=(1, 2, 2))
    image = reshape(Float32.(1:16), 1, 1, 4, 4, 1)
    down = vae_conv3d(spatial, image; spatial_padding=(0, 1, 0, 1))
    @test size(down) == (1, 1, 2, 2, 1)
    @test down[1, 1, 1, 1, 1] == sum(image[1, 1, 1:3, 1:3, 1])

    norm = VAERMSNorm(ones(Float32, 2))
    normalized = vae_rmsnorm(norm,
        reshape(Float32[3, 4], 2, 1, 1, 1, 1))
    @test sum(abs2, normalized) ≈ 2f0

    zero_conv(channels; kernel=3) = VAEConv3D(
        zeros(Float32, channels, channels, kernel, kernel, kernel),
        zeros(Float32, channels); padding=(kernel ÷ 2, kernel ÷ 2, kernel ÷ 2))
    residual = VAEResidualBlock(norm, zero_conv(2), norm,
        zero_conv(2), nothing)
    residual_input = randn(Float32, 2, 2, 3, 3, 1)
    @test vae_residual_forward(residual, residual_input) == residual_input

    qkv = VAEConv3D(randn(Float32, 6, 2, 1, 1, 1),
        randn(Float32, 6))
    projection = VAEConv3D(zeros(Float32, 2, 2, 1, 1, 1),
        zeros(Float32, 2))
    attention = VAEAttentionBlock(norm, qkv, projection)
    @test vae_attention_forward(attention, residual_input) == residual_input

    latents = randn(Float32, 16, 2, 3, 4)
    @test unscale_wan_latents(scale_wan_latents(latents)) ≈ latents
    half_latents = Float16.(latents)
    @test eltype(scale_wan_latents(half_latents)) === Float16
    @test unscale_wan_latents(scale_wan_latents(half_latents)) ≈ half_latents rtol=2f-3

    tiny_config = WanVAEConfig(base_channels=2, latent_channels=16,
        channel_multipliers=[1, 2], residual_blocks=1,
        temporal_downsample=[true])
    encoder = WanVAEEncoder(tiny_config; rng=Xoshiro(92))
    video = randn(Xoshiro(93), Float32, 3, 5, 4, 4, 1)
    encoded = wan_vae_encoder_forward(encoder, video)
    @test size(encoded) == (16, 3, 2, 2, 1)
    @test all(isfinite, encoded)
    @test size(encode_video(Wan21(), encoder, dropdims(video; dims=5))) ==
        (16, 3, 2, 2)

    i2v_video = dropdims(video; dims=5)
    image_features = randn(Xoshiro(931), Float32, 1280, 5)
    prepared = prepare_wan_i2v_conditioning(
        Wan21(variant=:i2v_14b_480p), encoder, i2v_video, image_features)
    @test size(prepared.latents) == (16, 3, 2, 2)
    @test size(prepared.conditioning_video) == (20, 3, 2, 2)
    @test prepared.image_features === image_features
    @test all(prepared.conditioning_video[1:4, 1, :, :] .== 1f0)
    @test all(prepared.conditioning_video[1:4, 2:end, :, :] .== 0f0)
    reference = zeros(Float32, size(i2v_video))
    reference[:, 1, :, :] .= i2v_video[:, 1, :, :]
    @test prepared.conditioning_video[5:end, :, :, :] ≈
        encode_video(Wan21(), encoder, reference)
    @test_throws ArgumentError prepare_wan_i2v_conditioning(
        Wan21(), encoder, i2v_video, image_features)
    @test_throws DimensionMismatch prepare_wan_i2v_conditioning(
        Wan21(variant=:i2v_14b_480p), encoder, i2v_video,
        zeros(Float32, 1279, 5))

    state = wan_vae_encoder_state_dict(encoder)
    mktempdir() do directory
        path = joinpath(directory, "tiny-vae.safetensors")
        write_safetensors(path, state)
        source = open_tensor_source(path)
        @test isempty(audit_state_dict(source,
            wan_vae_encoder_specs(tiny_config)))
        loaded = load_wan_vae_encoder(source, tiny_config)
        @test wan_vae_encoder_state_dict(loaded) == state
        @test wan_vae_encoder_forward(loaded, video) ≈ encoded
    end
    @test length(wan_vae_encoder_specs()) == 86

    decoder = WanVAEDecoder(tiny_config; rng=Xoshiro(94))
    decoded = wan_vae_decoder_forward(decoder, encoded)
    @test size(decoded) == size(video)
    @test all(isfinite, decoded)
    decoded_full =
        wan_vae_decoder_forward(decoder, encoded; streaming=false)
    @test decoded ≈ decoded_full rtol=2f-5 atol=2f-5
    decoder_state = wan_vae_decoder_state_dict(decoder)
    full_state = merge(copy(state), decoder_state)
    mktempdir() do directory
        path = joinpath(directory, "tiny-full-vae.safetensors")
        write_safetensors(path, full_state)
        source = open_tensor_source(path)
        @test isempty(audit_state_dict(source, wan_vae_specs(tiny_config)))
        loaded = load_wan_vae(source, tiny_config)
        @test wan_vae_encoder_forward(loaded.encoder, video) ≈ encoded
        @test wan_vae_decoder_forward(loaded.decoder, encoded) ≈ decoded
        @test size(decode_video(Wan21(), loaded, dropdims(encoded; dims=5))) ==
            (3, 5, 4, 4)
    end

    @test Reels.wan_vae_diffusers_key("encoder.conv1.weight") ==
          "encoder.conv_in.weight"
    @test Reels.wan_vae_diffusers_key(
        "encoder.downsamples.1.residual.3.gamma") ==
          "encoder.down_blocks.1.norm2.gamma"
    @test Reels.wan_vae_diffusers_key(
        "decoder.upsamples.4.shortcut.weight") ==
          "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"
    @test Reels.wan_vae_diffusers_key(
        "decoder.upsamples.7.time_conv.weight") ==
          "decoder.up_blocks.1.upsamplers.0.time_conv.weight"
    diffusers_specs = Reels.wan_vae_diffusers_specs()
    @test length(diffusers_specs) == 194
    @test length(unique(getproperty.(diffusers_specs, :source_key))) == 194

    diffusers_state = Dict(
        Reels.wan_vae_diffusers_key(key) => value
        for (key, value) in full_state)
    @test length(diffusers_state) == length(full_state)
    mktempdir() do directory
        path = joinpath(directory, "tiny-diffusers-vae.safetensors")
        write_safetensors(path, diffusers_state)
        source = open_tensor_source(path)
        @test isempty(audit_state_dict(
            source, Reels.wan_vae_diffusers_specs(tiny_config)))
        loaded = load_wan_vae(source, tiny_config)
        @test wan_vae_encoder_state_dict(loaded.encoder) == state
        @test wan_vae_decoder_state_dict(loaded.decoder) == decoder_state

        invalid_path = joinpath(directory,
            "tiny-diffusers-vae-unexpected.safetensors")
        invalid_state = merge(copy(diffusers_state),
            Dict("unexpected.weight" => zeros(Float32, 1)))
        write_safetensors(invalid_path, invalid_state)
        @test_throws ArgumentError load_wan_vae(invalid_path, tiny_config)
    end
    @test length(wan_vae_decoder_specs()) == 108
    @test length(wan_vae_specs()) == 194
end
