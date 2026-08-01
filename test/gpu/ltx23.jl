using CUDA

@testset "CUDA FP16 LTX-2.3 video LoRA optimizer step" begin
    rng = MersenneTwister(401)
    config = LTX23Config(video_only=true, video_heads=2, video_head_dim=4,
        video_channels=5, video_context_dim=6, layers=1,
        video_max_positions=(8, 16, 16))
    base = LTXVideoTransformer(config; rng=rng)
    cpu = inject_ltx23_lora(base;
        targets=["transformer_blocks.0.attn1.to_q.weight"],
        rank=2, alpha=2, dropout=0.25, rng=rng)
    model = move_to_device(cpu, :cuda, :fp16)
    transfer = array_transfer(:cuda, :fp16)
    latents = transfer(randn(rng, Float32, 5, 3, 1))
    context = transfer(randn(rng, Float32, 6, 2, 1))
    timesteps = transfer(Float32[0.4])
    target = transfer(randn(rng, Float32, 5, 3, 1))
    positions = CUDA.zeros(Float32, 3, 3, 1)
    parameters = [entry.value for entry in ltx23_lora_parameters(model)]
    before = map(Array, parameters)

    result = ltx23_lora_step!(
        model, AdamW(learning_rate=1f-2, weight_decay=0f0),
        AdamWState(parameters), latents, timesteps, context, positions,
        target; checkpoint_interval=1, loss_scale=128f0,
        dropout_seed=UInt64(13))
    CUDA.synchronize()

    @test isfinite(result.loss)
    @test result.gradient_norm > 0f0
    @test any(index -> Array(parameters[index]) != before[index],
              eachindex(parameters))
end

@testset "CUDA BF16 LTX-2.3 video LoRA" begin
    rng = MersenneTwister(402)
    config = LTX23Config(video_only=true, video_heads=2, video_head_dim=4,
        video_channels=5, video_context_dim=6, layers=1,
        video_max_positions=(8, 16, 16))
    base = LTXVideoTransformer(config; rng=rng)
    cpu = inject_ltx23_lora(base;
        targets=["transformer_blocks.0.attn1.to_q.weight"],
        rank=2, alpha=2, rng=rng)
    model = move_to_device(cpu, :cuda, :bf16)
    transfer = array_transfer(:cuda, :bf16)
    latents = transfer(randn(rng, Float32, 5, 3, 1))
    context = transfer(randn(rng, Float32, 6, 2, 1))
    timesteps = transfer(Float32[0.4])
    target = transfer(randn(rng, Float32, 5, 3, 1))
    positions = zeros(Float32, 3, 3, 1)
    shadow = move_to_device(model, Reels.float32_values)

    differentiated = ltx23_lora_loss_and_gradients(
        model, latents, timesteps, context, positions, target;
        checkpoint_interval=1, shadow_model=shadow)
    @test isfinite(differentiated.loss)
    @test all(parameter -> eltype(parameter) === BFloat16,
              differentiated.parameters)
    @test all(gradient -> all(isfinite, gradient),
              differentiated.gradients)
    state = AdamWState(differentiated.parameters)
    before = map(copy, state.master)
    update!(AdamW(learning_rate=1f-2, weight_decay=0f0), state,
            differentiated.parameters, differentiated.gradients)
    CUDA.synchronize()
    @test all(master -> eltype(master) === Float32, state.master)
    @test any(index -> state.master[index] != before[index],
              eachindex(before))
    repeated = ltx23_lora_loss_and_gradients(
        model, latents, timesteps, context, positions, target;
        checkpoint_interval=1, shadow_model=shadow)
    @test isfinite(repeated.loss)
    source_parameters = ltx23_lora_parameters(model)
    shadow_parameters = ltx23_lora_parameters(shadow)
    @test all(eachindex(source_parameters)) do index
        Array(shadow_parameters[index].value) ==
            Array(Reels.float32_values(source_parameters[index].value))
    end
end

@testset "LTX RoPE accepts device-resident positions" begin
    config = LTX23Config(video_only=true, video_heads=1, video_head_dim=6,
        video_channels=3, video_context_dim=4, layers=1,
        video_max_positions=(8, 16, 16))
    like = CUDA.zeros(Float32, 6, 1, 1)
    positions = CUDA.zeros(Float32, 3, 1, 1)
    cosine, sine = ltx_rope_frequencies(config, positions, like)
    @test cosine isa CUDA.CuArray
    @test sine isa CUDA.CuArray
    @test size(cosine) == (3, 1, 1, 1)
    @test all(isfinite, Array(cosine))
    @test all(isfinite, Array(sine))
end

@testset "CUDA BF16 LTX-2.3 text conditioning" begin
    rng = MersenneTwister(403)
    gemma_config = Gemma3TextConfig(vocab_size=32, hidden_size=8,
        intermediate_size=16, layers=2, heads=2, kv_heads=1,
        head_dim=4, max_length=4, sliding_window=4,
        sliding_window_pattern=2, query_pre_attention_scalar=4)
    connector_config = LTXTextConnectorConfig(gemma_hidden_size=8,
        gemma_hidden_layers=3, output_dim=8, heads=2, head_dim=4,
        layers=1, registers=2, max_position=16)
    gemma = move_to_device(
        Gemma3TextEncoder(gemma_config; rng=rng), :cuda, :bf16)
    connector = move_to_device(
        LTXTextConnector(connector_config; rng=rng), :cuda, :bf16)
    tokenized = TokenizedText(
        Int32[0 0; 2 0; 3 2; 4 5],
        Bool[false false; true false; true true; true true],
        [3, 2])
    encoded = ltx23_text_conditioner_forward(
        LTXTextConditioner(nothing, gemma, connector), tokenized)
    CUDA.synchronize()
    @test encoded.context isa CUDA.CuArray
    @test eltype(encoded.context) === BFloat16
    @test size(encoded.context) == (8, 4, 2)
    @test all(isfinite, encoded.context)
    @test encoded.mask == Bool[true true; true true; true false; false false]
end
