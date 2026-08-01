@testset "LTX-2.3 Gemma feature projection and video connector" begin
    official = ltx23_text_connector_config(Dict("transformer" => Dict(
        "num_attention_heads" => 32,
        "attention_head_dim" => 128,
        "connector_num_attention_heads" => 32,
        "connector_attention_head_dim" => 128,
        "connector_num_layers" => 8,
        "connector_num_learnable_registers" => 128,
        "connector_positional_embedding_max_pos" => [4096],
        "connector_apply_gated_attention" => true)))
    @test official.output_dim == 4096
    @test official.gemma_hidden_size * official.gemma_hidden_layers ==
        188160
    @test length(ltx23_text_connector_specs(official)) == 131
    official_specs = Dict(spec.source_key => spec.source_shape
        for spec in ltx23_text_connector_specs(official))
    @test official_specs[
        "text_embedding_projection.video_aggregate_embed.weight"] ==
        [4096, 188160]
    @test official_specs[
        "model.diffusion_model.video_embeddings_connector.learnable_registers"] ==
        [128, 4096]
    @test official_specs[
        "model.diffusion_model.video_embeddings_connector." *
        "transformer_1d_blocks.7.attn1.to_gate_logits.weight"] ==
        [32, 4096]

    config = LTXTextConnectorConfig(gemma_hidden_size=4,
        gemma_hidden_layers=3, output_dim=8, heads=2, head_dim=4,
        layers=2, registers=2, max_position=16,
        gated_attention=true)
    rng = Xoshiro(240)
    hidden = [randn(rng, Float32, 4, 4, 1) for _ in 1:3]
    mask = reshape(Bool[false, true, true, true], 4, 1)
    features = ltx23_gemma_features(hidden, mask, config)
    @test size(features) == (12, 4, 1)
    @test all(iszero, features[:, 1, 1])
    for token in 2:4, layer in 0:2
        # PyTorch flattens [hidden,layer] with layer varying fastest.
        section = view(features, layer+1:3:12, token, 1)
        @test sum(abs2, section) ≈ 8f0 rtol=1f-5
    end

    connector = LTXTextConnector(config; rng=rng)
    encoded = ltx23_text_connector_forward(connector, hidden, mask)
    @test size(encoded) == (8, 4, 1)
    @test all(isfinite, encoded)
    @test all(isapprox(sum(abs2, encoded[:, token, 1]), 8f0;
                       rtol=2f-5) for token in 1:4)

    gemma_config = Gemma3TextConfig(vocab_size=16, hidden_size=4,
        intermediate_size=8, layers=2, heads=2, kv_heads=1,
        head_dim=2, max_length=4, sliding_window=4,
        sliding_window_pattern=2, query_pre_attention_scalar=2)
    gemma = Gemma3TextEncoder(gemma_config; rng=rng)
    tokenized = TokenizedText(
        reshape(Int32[0, 0, 2, 3], 4, 1),
        reshape(Bool[false, false, true, true], 4, 1), [2])
    conditioned = ltx23_text_conditioner_forward(
        LTXTextConditioner(nothing, gemma, connector), tokenized)
    @test size(conditioned.context) == (8, 4, 1)
    @test conditioned.mask[:, 1] == Bool[true, true, false, false]
    @test conditioned.lengths == [2]
    @test all(isfinite, conditioned.context)

    bad_gemma = deepcopy(gemma)
    fill!(bad_gemma.embedding, Float32(NaN))
    @test_throws ArgumentError ltx23_text_conditioner_forward(
        LTXTextConditioner(nothing, bad_gemma, connector), tokenized)

    mktempdir() do directory
        media = joinpath(directory, "source.bin")
        write(media, UInt8[1, 2, 3])
        sample = VideoSample("sample", media, "tiny caption")
        bucket = BucketAssignment(32, 32, 9, 25, 32, 32, 0, 0)
        identity = PreprocessIdentity(model_family="ltx23",
            model_checkpoint="transformer", text_encoder_checkpoint="gemma",
            vae_checkpoint="vae", vae_scale="native", dtype="fp32")
        latent_volume = randn(rng, Float32, 2, 2, 1, 2)
        cached = write_ltx23_preprocess_cache(directory, sample, bucket,
            identity, latent_volume, conditioned)
        @test cached.created
        @test ltx23_cache_is_valid(cached.path, cached.key)
        loaded = load_ltx23_preprocess_cache(cached.path;
                                             expected_key=cached.key)
        @test size(loaded.latents) == (2, 4)
        @test size(loaded.positions) == (3, 4)
        @test size(loaded.text_context) == (8, 4)
        @test loaded.metadata["caption_token_length"] == "2"
        reused = write_ltx23_preprocess_cache(directory, sample, bucket,
            identity, latent_volume, conditioned)
        @test !reused.created
    end

    mktempdir() do directory
        specs = ltx23_text_connector_specs(config)
        state = Dict{String,AbstractArray}(
            spec.source_key => zeros(Float32, spec.source_shape...)
            for spec in specs)
        state[
            "model.diffusion_model.video_embeddings_connector.learnable_registers"] =
            zeros(BFloat16, config.registers, config.output_dim)
        path = joinpath(directory, "tiny-connector.safetensors")
        write_safetensors(path, state)
        source = open_tensor_source(path)
        @test isempty(audit_state_dict(source, specs))
        loaded = load_ltx23_text_connector(source, config; strict=true)
        output = ltx23_text_connector_forward(loaded, hidden, mask)
        @test output == zeros(Float32, 8, 4, 1)
    end

    @test_throws DimensionMismatch ltx23_gemma_features(hidden[1:2],
        mask, config)
    @test_throws ArgumentError LTXTextConnectorConfig(output_dim=7,
        heads=2, head_dim=4)
end
