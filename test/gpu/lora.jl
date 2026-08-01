using CUDA

@testset "CUDA bounded-memory tiled attention" begin
    rng = MersenneTwister(38)
    q = randn(rng, Float32, 8, 2, 17, 1)
    k = randn(rng, Float32, 8, 2, 19, 1)
    v = randn(rng, Float32, 8, 2, 19, 1)
    dy = randn(rng, Float32, 8, 2, 17, 1)
    reference = reference_attention(q, k, v)
    expected_gradient = attention_backward(q, k, v, dy, reference.cache)
    actual = memory_efficient_attention(
        CuArray(q), CuArray(k), CuArray(v); query_block=5, key_block=7)
    gradient = memory_efficient_attention_backward(
        CuArray(q), CuArray(k), CuArray(v), CuArray(dy), actual.cache)
    CUDA.synchronize()
    @test Array(actual.output) ≈ reference.output rtol=3f-5 atol=3f-5
    @test Array(gradient.q) ≈ expected_gradient.q rtol=4f-5 atol=4f-5
    @test Array(gradient.k) ≈ expected_gradient.k rtol=4f-5 atol=4f-5
    @test Array(gradient.v) ≈ expected_gradient.v rtol=4f-5 atol=4f-5
end

@testset "CUDA tiny UMT5 encoder" begin
    rng = MersenneTwister(39)
    config = UMT5Config(vocab_size=13, hidden_size=4, attention_size=4,
        ffn_size=6, heads=2, layers=1, buckets=8)
    dense(out, input) = DenseLayer(
        randn(rng, Float32, out, input) ./ sqrt(Float32(input)), nothing)
    attention = T5Attention(dense(4, 4), dense(4, 4), dense(4, 4),
        dense(4, 4), 2)
    feed_forward = T5FeedForward(dense(6, 4), dense(6, 4), dense(4, 6))
    block = T5EncoderBlock(RMSNorm(4), attention, RMSNorm(4),
        feed_forward, randn(rng, Float32, 2, 8))
    cpu = UMT5Encoder(config, randn(rng, Float32, 4, 13), [block],
        RMSNorm(4), nothing)
    ids = Int32[0 3; 1 4; 2 5]
    mask = BitMatrix(Bool[1 1; 1 1; 0 1])
    expected = umt5_forward(cpu, ids; mask=mask)
    gpu = move_to_device(cpu, :cuda)
    actual_device = umt5_forward(gpu, ids; mask=CuArray(mask))
    CUDA.synchronize()
    @test actual_device isa CuArray
    @test Array(actual_device) ≈ expected rtol=3f-4 atol=3f-4
end

@testset "CUDA tiny Wan VAE" begin
    rng = MersenneTwister(40)
    config = WanVAEConfig(base_channels=2, latent_channels=16,
        channel_multipliers=[1, 2], residual_blocks=1,
        temporal_downsample=[true])
    cpu = WanVAEEncoder(config; rng=rng)
    video = randn(rng, Float32, 3, 5, 8, 8, 1)
    expected = wan_vae_encoder_forward(cpu, video)
    gpu = move_to_device(cpu, :cuda)
    actual = wan_vae_encoder_forward(gpu, CuArray(video))
    CUDA.synchronize()
    @test actual isa CuArray
    @test Array(actual) ≈ expected rtol=5f-4 atol=5f-4
end

@testset "CUDA BF16 Wan LoRA with FP32 optimizer masters" begin
    rng = MersenneTwister(401)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=8, ffn_size=16, frequency_size=4,
        text_size=6, output_channels=1, heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    cpu = inject_wan_lora(base;
        targets=["blocks.0.self_attn.q.weight"], rank=2, alpha=2, rng=rng)
    model = move_to_device(cpu, :cuda, :bf16)
    convert_input(values) = array_transfer(:cuda, :bf16)(values)
    video = convert_input(randn(rng, Float32, 1, 1, 2, 4, 1))
    text = convert_input(randn(rng, Float32, 6, 2, 1))
    timesteps = convert_input(Float32[0.4])
    target = convert_input(randn(rng, Float32, size(video)))

    differentiated = wan_lora_loss_and_gradients(
        model, video, timesteps, text, target; checkpoint_interval=1)
    parameters = differentiated.parameters
    @test all(parameter -> eltype(parameter) === BFloat16, parameters)
    @test all(gradient -> all(isfinite, gradient),
              differentiated.gradients)
    state = AdamWState(parameters)
    @test all(index -> Array(state.master[index]) ==
                       Array(Reels.float32_values(parameters[index])),
              eachindex(parameters))
    before = map(copy, state.master)
    update!(AdamW(learning_rate=1f-2, weight_decay=0f0), state,
            parameters, differentiated.gradients)
    CUDA.synchronize()
    @test all(master -> eltype(master) === Float32, state.master)
    @test any(index -> state.master[index] != before[index],
              eachindex(state.master))
    @test all(index -> isapprox(Array(Reels.float32_values(parameters[index])),
                                Array(state.master[index]);
                                rtol=8f-3, atol=8f-3),
              eachindex(parameters))
    mktempdir() do directory
        checkpoint = joinpath(directory, "bf16.reels")
        training_state = TrainingState(
            1, 1, Xoshiro(402), state)
        save_checkpoint(checkpoint, training_state, parameters)
        restored = load_checkpoint(checkpoint)
        @test all(index -> restored.params[index] ==
                           Array(Reels.float32_values(parameters[index])),
                  eachindex(parameters))
        @test restored.state.optimizer.master ==
              Array.(state.master)
    end
end

@testset "CUDA LoRA forward, backward, and AdamW" begin
    @test CUDA.functional()
    CUDA.device!(0)

    rng = MersenneTwister(41)
    weight = randn(rng, Float32, 7, 5)
    A = randn(rng, Float32, 3, 5)
    B = randn(rng, Float32, 7, 3)
    x = randn(rng, Float32, 5, 4)
    dy = randn(rng, Float32, 7, 4)

    cpu = LoRALinear(weight, nothing, A, B, 6f0, 0f0, true)
    gpu = LoRALinear(CuArray(weight), nothing, CuArray(A), CuArray(B), 6f0, 0f0, true)

    expected_y = lora_forward(cpu, x)
    expected_grad = lora_backward(cpu, x, dy)
    actual_y = Array(lora_forward(gpu, CuArray(x)))
    actual_grad = lora_backward(gpu, CuArray(x), CuArray(dy))
    CUDA.synchronize()

    @test actual_y ≈ expected_y rtol=2f-5 atol=2f-5
    @test Array(actual_grad.dx) ≈ expected_grad.dx rtol=2f-5 atol=2f-5
    @test Array(actual_grad.A) ≈ expected_grad.A rtol=2f-5 atol=2f-5
    @test Array(actual_grad.B) ≈ expected_grad.B rtol=2f-5 atol=2f-5

    params = [gpu.A, gpu.B]
    grads = [actual_grad.A, actual_grad.B]
    before = map(p -> Array(p), params)
    state = AdamWState(params)
    update!(AdamW(learning_rate=1f-3), state, params, grads)
    CUDA.synchronize()

    @test state.step == 1
    @test all(s -> s isa CuArray, state.m)
    @test all(s -> s isa CuArray, state.v)
    @test all(p -> all(isfinite, Array(p)), params)
    @test any(i -> Array(params[i]) != before[i], eachindex(params))
end

@testset "CUDA tiny Wan full transformer" begin
    rng = MersenneTwister(43)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=5,
        input_channels=2, hidden_size=12, ffn_size=24, frequency_size=8,
        text_size=10, output_channels=2, heads=3, layers=2)
    cpu = WanTransformer(config; rng=rng)
    cpu.head.weight .= randn(rng, Float32, size(cpu.head.weight))
    cpu.head.bias .= randn(rng, Float32, size(cpu.head.bias))
    video = randn(rng, Float32, 2, 2, 4, 4, 2)
    text = randn(rng, Float32, 10, 5, 2)
    timesteps = Float32[0.2, 0.7]

    expected = wan_transformer_forward(cpu, video, timesteps, text)
    gpu = move_to_device(cpu, :cuda)
    actual_device = wan_transformer_forward(
        gpu, CuArray(video), CuArray(timesteps), CuArray(text))
    CUDA.synchronize()
    actual = Array(actual_device)

    @test actual_device isa CuArray
    @test size(actual) == size(expected)
    @test all(isfinite, actual)
    @test actual ≈ expected rtol=2f-4 atol=2f-4

    restored = move_to_device(gpu, :cpu)
    @test restored.patch_weight isa Array
    @test wan_transformer_forward(restored, video, timesteps, text) ≈ expected
end

@testset "CUDA Wan LoRA model" begin
    rng = MersenneTwister(54)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=5,
        input_channels=2, hidden_size=12, ffn_size=24, frequency_size=8,
        text_size=10, output_channels=2, heads=3, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    cpu = inject_wan_lora(base;
        targets=lora_targets(Wan21(), base, :attention), rank=2, alpha=4,
        rng=rng)
    for entry in wan_lora_layers(cpu)
        entry.layer.B .= randn(rng, Float32, size(entry.layer.B)) .* 0.01f0
    end
    video = randn(rng, Float32, 2, 2, 4, 4, 1)
    text = randn(rng, Float32, 10, 5, 1)
    timesteps = Float32[0.6]
    expected = wan_transformer_forward(cpu, video, timesteps, text)

    gpu = move_to_device(cpu, :cuda)
    @test all(parameter -> parameter.value isa CuArray,
              wan_lora_parameters(gpu))
    actual = Array(wan_transformer_forward(
        gpu, CuArray(video), CuArray(timesteps), CuArray(text)))
    CUDA.synchronize()
    @test actual ≈ expected rtol=3f-4 atol=3f-4
end

@testset "CUDA Wan LoRA training step" begin
    rng = MersenneTwister(62)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
        text_size=5, output_channels=1, heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    cpu = inject_wan_lora(base;
        targets=["blocks.0.self_attn.q.weight"], rank=1, alpha=1, rng=rng)
    model = move_to_device(cpu, :cuda)
    video = CuArray(randn(rng, Float32, 1, 1, 2, 4, 1))
    text = CuArray(randn(rng, Float32, 5, 2, 1))
    timesteps = CuArray(Float32[0.5])
    target = CuArray(randn(rng, Float32, size(video)))
    parameters = [entry.value for entry in wan_lora_parameters(model)]
    before = map(parameter -> Array(parameter), parameters)
    state = AdamWState(parameters)

    result = wan_lora_step!(model,
        AdamW(learning_rate=1f-2, weight_decay=0f0), state,
        video, timesteps, text, target)
    CUDA.synchronize()

    @test isfinite(result.loss)
    @test result.gradient_norm > 0
    @test state.step == 1
    @test all(parameter -> all(isfinite, Array(parameter)), parameters)
    @test any(i -> Array(parameters[i]) != before[i], eachindex(parameters))
end

@testset "CUDA Wan I2V FP16 LoRA training step" begin
    rng = MersenneTwister(63)
    config = Wan21Config(
        variant=:i2v_test, model_type=:i2v,
        patch_size=(1, 2, 2), text_length=2,
        input_channels=36, hidden_size=8, ffn_size=16,
        frequency_size=4, text_size=6, output_channels=16,
        heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    cpu = inject_wan_lora(base;
        targets=["blocks.0.cross_attn.k_img.weight"],
        rank=1, alpha=1, dropout=0.25, rng=rng)
    model = move_to_device(cpu, :cuda, :fp16)
    transfer = array_transfer(:cuda, :fp16)
    video = transfer(randn(rng, Float32, 16, 1, 2, 4, 1))
    conditioning = transfer(randn(rng, Float32, 20, 1, 2, 4, 1))
    text = transfer(randn(rng, Float32, 6, 2, 1))
    image = transfer(randn(rng, Float32, 1280, 3, 1))
    timesteps = transfer(Float32[0.5])
    target = transfer(randn(rng, Float32, size(video)))
    parameters = [entry.value for entry in wan_lora_parameters(model)]
    before = map(Array, parameters)

    result = wan_lora_step!(
        model, AdamW(learning_rate=1f-2, weight_decay=0f0),
        AdamWState(parameters), video, timesteps, text, target;
        checkpoint_interval=1, loss_scale=128f0,
        dropout_seed=UInt64(12),
        conditioning_video=conditioning, image_features=image)
    CUDA.synchronize()

    @test isfinite(result.loss)
    @test result.gradient_norm > 0f0
    @test any(index -> Array(parameters[index]) != before[index],
              eachindex(parameters))
end

@testset "CUDA Wan latent training loop" begin
    rng = MersenneTwister(74)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
        text_size=5, output_channels=1, heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    cpu = inject_wan_lora(base;
        targets=["blocks.0.self_attn.v.weight"], rank=1, alpha=1, rng=rng)
    model = move_to_device(cpu, :cuda)
    latents = CuArray(randn(rng, Float32, 1, 1, 2, 4, 1))
    context = CuArray(randn(rng, Float32, 5, 2, 1))
    noise = CuArray(randn(rng, Float32, size(latents)))
    timesteps = CuArray(Float32[0.4])
    batch = WanLatentBatch(latents, context; noise=noise,
                           timesteps=timesteps)
    training = TrainingConfig(steps=1, gradient_accumulation=2,
        learning_rate=0.01f0, weight_decay=0f0,
        max_gradient_norm=1f0, seed=75)

    mktempdir() do dir
        job = WanTrainingJob(model=model, batch=_ -> batch,
            training=training, output_dir=dir,
            checkpoint=CheckpointConfig(every_steps=1, keep_last=1),
            base_model="tiny-cuda-test")
        result = train!(job)
        CUDA.synchronize()
        @test result.state.step == 1
        @test result.state.micro_step == 2
        @test length(result.losses) == 1
        @test all(isfinite, result.losses)
        @test isfile(joinpath(dir, "checkpoint-1.reels"))
        @test isfile(joinpath(dir, "adapter-1.safetensors"))
        @test isfile(result.adapter)
        restored = load_checkpoint(joinpath(dir, "checkpoint-1.reels"))
        @test restored.state.step == 1
        @test length(restored.params) == 2
    end
end

@testset "CUDA cached batch provider" begin
    mktempdir() do dir
        entries = String[]
        for index in 1:2
            key = string(index)^64
            path = cache_entry_path(dir, key)
            write_preprocess_cache(path, key,
                fill(Float32(index), 1, 1, 2, 2),
                fill(Float32(index), 5, 2))
            push!(entries, path)
        end
        provider = CachedBatchProvider(entries; batch_size=2, device=:cuda)
        batch = provider(Xoshiro(81))
        CUDA.synchronize()
        @test batch.latents isa CuArray
        @test batch.text_context isa CuArray
        @test size(batch.latents) == (1, 1, 2, 2, 2)
        @test size(batch.text_context) == (5, 2, 2)
    end
end
