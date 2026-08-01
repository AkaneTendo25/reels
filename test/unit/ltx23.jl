@testset "LTX-2.3 latent patchifier and positions" begin
    latents = reshape(Float32.(1:(2 * 2 * 4 * 6)), 2, 2, 4, 6, 1)
    tokens = ltx23_patchify_latents(latents; patch_size=(1, 2, 3))
    @test size(tokens) == (12, 8, 1)
    @test ltx23_unpatchify_latents(tokens, size(latents);
        patch_size=(1, 2, 3)) == latents

    unit_tokens = ltx23_patchify_latents(latents)
    @test unit_tokens[:, 1, 1] == latents[:, 1, 1, 1, 1]
    @test unit_tokens[:, 2, 1] == latents[:, 1, 1, 2, 1]
    @test unit_tokens[:, 7, 1] == latents[:, 1, 2, 1, 1]

    bounds = ltx23_patch_bounds(2, 2, 2; batch=2)
    @test size(bounds) == (3, 8, 2, 2)
    @test bounds[:, 1, :, 1] == Int32[0 1; 0 1; 0 1]
    @test bounds[:, 2, :, 1] == Int32[0 1; 0 1; 1 2]
    @test bounds[:, 5, :, 1] == Int32[1 2; 0 1; 0 1]
    @test bounds[:, :, :, 1] == bounds[:, :, :, 2]

    pixel = ltx23_pixel_bounds(bounds)
    @test pixel[1, 1, :, 1] == Float32[0, 1]
    @test pixel[1, 5, :, 1] == Float32[1, 9]
    @test pixel[2, 1, :, 1] == Float32[0, 32]
    @test pixel[3, 2, :, 1] == Float32[32, 64]

    positions = ltx23_patch_positions(2, 2, 2; batch=2, fps=25)
    @test size(positions) == (3, 8, 2)
    @test positions[:, 1, 1] == Float32[0.5 / 25, 16, 16]
    @test positions[:, 2, 1] == Float32[0.5 / 25, 16, 48]
    @test positions[:, 5, 1] == Float32[5 / 25, 16, 16]
    @test positions[:, :, 1] == positions[:, :, 2]

    @test_throws DimensionMismatch ltx23_patchify_latents(latents;
        patch_size=(1, 3, 1))
    @test_throws ArgumentError ltx23_patch_positions(1, 1, 1; fps=0)
end

@testset "LTX-2.3 configuration, inventory, and backend contract" begin
    tiny = LTX23Config(video_only=true, video_heads=2, video_head_dim=4,
        video_channels=8, video_context_dim=12, audio_heads=2,
        audio_head_dim=3, layers=2)
    @test ltx23_video_dim(tiny) == 8
    @test ltx23_audio_dim(tiny) == 6
    specs = ltx23_transformer_specs(tiny)
    @test length(specs) == 61
    @test length(unique(spec.source_key for spec in specs)) == length(specs)
    @test only(filter(s -> s.source_key ==
        "transformer_blocks.1.attn2.to_k.weight", specs)).source_shape == [8, 12]
    @test only(filter(s -> s.source_key ==
        "transformer_blocks.0.ff.net.0.proj.weight", specs)).source_shape == [32, 8]

    prefixed = ltx23_transformer_specs(tiny;
        checkpoint_prefix="model.diffusion_model.")
    @test all(startswith(spec.source_key, "model.diffusion_model.")
              for spec in prefixed)

    parsed = ltx23_config(Dict("transformer" => Dict(
        "num_attention_heads" => 4,
        "attention_head_dim" => 8,
        "num_layers" => 3,
        "cross_attention_adaln" => true,
        "positional_embedding_max_pos" => [24, 1024, 768])))
    @test (ltx23_video_dim(parsed), parsed.layers,
           parsed.cross_attention_adaln) == (32, 3, true)
    @test parsed.video_max_positions == (24, 1024, 768)

    backend = LTX23(video_only=true)
    @test model_family(backend) == :ltx23
    @test load_config(backend).video_only
    @test length(lora_targets(backend, nothing, :attention_and_ffn)) == 2
    @test export_mapping(backend)["prefix"] == "diffusion_model."
    @test_throws ArgumentError lora_targets(backend, nothing, :unknown)
end

@testset "LTX-2.3 LoRA training and adapter round-trip" begin
    rng = MersenneTwister(231)
    config = LTX23Config(video_only=true, video_heads=2, video_head_dim=4,
        video_channels=5, video_context_dim=6, layers=1,
        video_max_positions=(8, 16, 16))
    base = LTXVideoTransformer(config; rng=rng)
    model = inject_ltx23_lora(base;
        targets=["transformer_blocks.0.attn1.to_q.weight"],
        rank=2, alpha=2, dropout=0.25, train_bias=true,
        rank_overrides=Dict("transformer_blocks.0.attn1.to_q" => 1),
        alpha_overrides=Dict(
            "transformer_blocks.0.attn1.to_q.weight" => 3f0),
        rng=rng)
    @test length(ltx23_lora_layers(model)) == 1
    @test length(ltx23_lora_parameters(model)) == 3
    @test size(only(ltx23_lora_layers(model)).layer.A, 1) == 1
    @test only(ltx23_lora_layers(model)).layer.alpha == 3f0
    @test only(ltx23_lora_layers(model)).layer.train_bias
    @test startswith(first(ltx23_lora_parameters(model)).name,
                     "diffusion_model.transformer_blocks.0")

    latents = randn(rng, Float32, 5, 3, 1)
    context = randn(rng, Float32, 6, 2, 1)
    positions = zeros(Float32, 3, 3, 1)
    target = randn(rng, Float32, size(latents))
    differentiated = ltx23_lora_loss_and_gradients(
        model, latents, Float32[0.4], context, positions, target;
        checkpoint_interval=1, dropout_seed=UInt64(11))
    @test isfinite(differentiated.loss)
    @test all(gradient -> all(isfinite, gradient),
              differentiated.gradients)
    loss_scaled = ltx23_lora_loss_and_gradients(
        model, latents, Float32[0.4], context, positions, target;
        checkpoint_interval=1, loss_scale=128f0,
        dropout_seed=UInt64(11))
    @test loss_scaled.loss ≈ differentiated.loss rtol=2f-6 atol=2f-6
    @test all(eachindex(differentiated.gradients)) do index
        loss_scaled.gradients[index] ≈ differentiated.gradients[index]
    end
    state = AdamWState(differentiated.parameters)
    before = map(copy, differentiated.parameters)
    step = ltx23_lora_step!(model,
        AdamW(learning_rate=1f-2, weight_decay=0f0), state,
        latents, Float32[0.4], context, positions, target;
        checkpoint_interval=1, loss_scale=128f0,
        dropout_seed=UInt64(11))
    @test isfinite(step.loss)
    @test any(index -> differentiated.parameters[index] != before[index],
              eachindex(before))
    comparison = ltx23_validation_comparison(
        model, randn(rng, Float32, size(latents)),
        context, positions; steps=2)
    @test comparison.mean_absolute_delta > 0
    @test comparison.enabled != comparison.disabled
    @test all(entry -> entry.layer.enabled, ltx23_lora_layers(model))

    mktempdir() do dir
        path = joinpath(dir, "ltx-lora.safetensors")
        save_ltx23_lora(path, model; base_model="ltx-2.3-22b-dev")
        header = inspect_safetensors(path)
        @test header.metadata["format"] == "reels-ltx23-lora-v2"
        @test header.metadata["scaling_baked_into_lora_B"] == "true"
        restored = inject_ltx23_lora(base;
            targets=["transformer_blocks.0.attn1.to_q.weight"],
            rank=1, alpha=3, train_bias=true,
            rng=MersenneTwister(999))
        load_ltx23_lora!(restored, path)
        @test ltx23_lora_state_dict(restored) ==
            ltx23_lora_state_dict(model)
    end

    @test_throws ArgumentError inject_ltx23_lora(base;
        targets=["transformer_blocks.0.attn1.to_q.weight"], rank=1,
        alpha_overrides=Dict("transformer_blocks.9.attn1.to_q" => 2f0))

    scaled = inject_ltx23_lora(base;
        targets=["transformer_blocks.0.attn1.to_q.weight"],
        rank=2, alpha=4, rng=MersenneTwister(1001))
    only(ltx23_lora_layers(scaled)).layer.B .= 3f0
    exported = ltx23_lora_state_dict(scaled)
    @test exported[
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight"] ==
        fill(6f0, size(only(ltx23_lora_layers(scaled)).layer.B))
end

@testset "LTX-2.3 native video transformer" begin
    rng = MersenneTwister(230)
    config = LTX23Config(video_only=true, video_heads=2, video_head_dim=4,
        video_channels=5, video_context_dim=6, layers=2,
        video_max_positions=(8, 16, 16))
    model = LTXVideoTransformer(config; rng=rng)
    latents = randn(rng, Float32, 5, 4, 2)
    context = randn(rng, Float32, 6, 3, 2)
    positions = zeros(Float32, 3, 4, 2)
    positions[1, :, :] .= Float32[0, 1, 2, 3]
    positions[2, :, :] .= Float32[0, 0, 1, 1]
    positions[3, :, :] .= Float32[0, 1, 0, 1]
    timesteps = Float32[0.2, 0.7]

    embedding = ltx_timestep_embedding(Float32[0], 8)
    @test embedding[:, 1] == Float32[1, 1, 1, 1, 0, 0, 0, 0]
    frequencies = ltx_rope_frequencies(config, positions, latents)
    @test size(frequencies[1]) == (2, 2, 4, 2)
    q = randn(rng, Float32, 4, 2, 4, 2)
    @test sum(abs2, ltx_apply_rope(q, frequencies)) ≈ sum(abs2, q) rtol=2f-6

    ordering_config = LTX23Config(video_only=true, video_heads=1,
        video_head_dim=12, video_channels=2, video_context_dim=3, layers=1,
        rope_theta=100f0, video_max_positions=(10, 10, 10))
    ordering_positions = reshape(Float32[1, 2, 3], 3, 1, 1)
    ordering = ltx_rope_frequencies(ordering_config, ordering_positions,
        zeros(Float32, 2, 1, 1))
    indices = Float32[1, 100] .* Float32(pi / 2)
    expected_angles = Float32[
        indices[1] * (2f0 * 1f0 / 10f0 - 1f0),
        indices[1] * (2f0 * 2f0 / 10f0 - 1f0),
        indices[1] * (2f0 * 3f0 / 10f0 - 1f0),
        indices[2] * (2f0 * 1f0 / 10f0 - 1f0),
        indices[2] * (2f0 * 2f0 / 10f0 - 1f0),
        indices[2] * (2f0 * 3f0 / 10f0 - 1f0),
    ]
    @test vec(ordering[1]) ≈ cos.(expected_angles) rtol=2f-6 atol=2f-6
    @test vec(ordering[2]) ≈ sin.(expected_angles) rtol=2f-6 atol=2f-6

    ordinary = ltx_video_transformer_forward(
        model, latents, timesteps, context, positions)
    checkpointed = ltx_video_transformer_forward(
        model, latents, timesteps, context, positions;
        checkpoint_interval=1)
    @test size(ordinary) == size(latents)
    @test all(isfinite, ordinary)
    @test checkpointed ≈ ordinary rtol=2f-6 atol=2f-6

    gated_config = LTX23Config(video_only=true, video_heads=2,
        video_head_dim=4, video_channels=5, video_context_dim=6,
        layers=1, gated_attention=true,
        video_max_positions=(8, 16, 16))
    gated = LTXVideoTransformer(gated_config; rng=MersenneTwister(233))
    @test gated.blocks[1].self_attention.gate !== nothing
    @test count(spec -> occursin("to_gate_logits", spec.source_key),
                ltx23_transformer_specs(gated_config)) == 4
    gated_output = ltx_video_transformer_forward(
        gated, latents, timesteps, context, positions)
    @test size(gated_output) == size(latents)
    @test all(isfinite, gated_output)

    cross_config = LTX23Config(video_only=true, video_heads=2,
        video_head_dim=4, video_channels=5, video_context_dim=8,
        layers=1, cross_attention_adaln=true, gated_attention=true,
        video_max_positions=(8, 16, 16))
    cross_model = LTXVideoTransformer(cross_config;
        rng=MersenneTwister(234))
    cross_context = randn(MersenneTwister(235), Float32, 8, 3, 2)
    cross_output = ltx_video_transformer_forward(cross_model,
        latents, timesteps, cross_context, positions; sigma=timesteps)
    @test cross_model.prompt_adaln !== nothing
    @test cross_model.blocks[1].prompt_scale_shift_table !== nothing
    @test size(cross_output) == size(latents)
    @test all(isfinite, cross_output)
    @test_throws ArgumentError LTXVideoTransformer(LTX23Config(
        video_only=true, video_heads=2, video_head_dim=4,
        video_context_dim=6, layers=1, cross_attention_adaln=true))

    mktempdir() do dir
        tensors = Dict{String,AbstractArray}()
        for spec in ltx23_transformer_specs(config)
            tensors[spec.source_key] = zeros(Float32, spec.source_shape...)
        end
        tensors["patchify_proj.weight"] .=
            reshape(Float32.(1:length(tensors["patchify_proj.weight"])),
                    size(tensors["patchify_proj.weight"])) ./ 100
        path = joinpath(dir, "ltx-tiny.safetensors")
        write_safetensors(path, tensors;
            metadata=Dict("config" => """{"transformer":{
                "num_attention_heads":2,"attention_head_dim":4,
                "in_channels":5,"cross_attention_dim":6,"num_layers":2,
                "positional_embedding_max_pos":[8,16,16]}}"""))
        parsed = ltx23_config(path; video_only=true)
        @test (ltx23_video_dim(parsed), parsed.layers) == (8, 2)
        loaded = load_ltx23_video_transformer(path, config)
        @test loaded.patchify_proj.weight ==
            tensors["patchify_proj.weight"]
        @test all(iszero, loaded.blocks[2].scale_shift_table)
    end
end
