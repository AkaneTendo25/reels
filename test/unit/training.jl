@testset "optimizer, training, and exact resume" begin
    @test learning_rate(1f0, SchedulerConfig(kind=:linear), 5, 10) ≈ 0.5f0
    backend = SyntheticBackend(3, 2)
    teacher = Float32[2 0 0; 0 -1 0.5]
    makejob(dir, steps, seed=12) = begin
        model = load_transformer(backend; rank=2, rng=Xoshiro(99))
        batch = rng -> synthetic_batch(rng, 3, 2, 8; teacher=teacher)
        cfg = TrainingConfig(steps=steps, gradient_accumulation=2,
            learning_rate=0.05f0, weight_decay=0f0, max_gradient_norm=10f0,
            seed=seed)
        TrainingJob(backend=backend, model=model, batch=batch, training=cfg,
            output_dir=dir, checkpoint=CheckpointConfig(every_steps=5))
    end
    mktempdir() do dir
        full = makejob(joinpath(dir, "full"), 10)
        full_result = train!(full)
        @test full_result.losses[end] < full_result.losses[1]

        partial = makejob(joinpath(dir, "partial"), 5)
        train!(partial)
        resumed = makejob(joinpath(dir, "resumed"), 10)
        train!(resumed; resume_from=joinpath(dir, "partial", "checkpoint-5.reels"))
        @test resumed.model.projection.A == full.model.projection.A
        @test resumed.model.projection.B == full.model.projection.B
    end
end

@testset "distributed checkpoint resume identity" begin
    @test isnothing(Reels._validate_resume_world_size(
        Dict("distributed.world_size" => "2"), 2))
    @test isnothing(Reels._validate_resume_world_size(Dict{String,String}(), 1))
    @test isnothing(Reels._validate_resume_world_size(
        Dict("model_family" => "wan21"), 1))
    @test_throws ArgumentError Reels._validate_resume_world_size(
        Dict("distributed.world_size" => "2"), 1)
    @test_throws ArgumentError Reels._validate_resume_world_size(
        Dict("distributed.world_size" => "invalid"), 2)
end

@testset "low-VRAM checkpoint resume identity" begin
    metadata = Dict(
        "low_vram.frozen_weight_quantization" => "int8",
        "low_vram.cpu_offload" => "true")
    config = LowVRAMConfig(
        frozen_weight_quantization=:int8, cpu_offload=true)
    @test isnothing(Reels._validate_resume_low_vram(metadata, config))
    @test isnothing(Reels._validate_resume_low_vram(
        Dict{String,String}(), LowVRAMConfig()))
    @test_throws ArgumentError Reels._validate_resume_low_vram(
        metadata, LowVRAMConfig(
            frozen_weight_quantization=:none, cpu_offload=true))
    @test_throws ArgumentError Reels._validate_resume_low_vram(
        metadata, LowVRAMConfig(
            frozen_weight_quantization=:int8, cpu_offload=false))
end

@testset "non-finite training steps fail before update" begin
    @test isnothing(Reels._assert_finite_training_step(
        "test", 1, 1f0, 2f0))
    @test_throws DomainError Reels._assert_finite_training_step(
        "test", 2, NaN32, 1f0)
    @test_throws DomainError Reels._assert_finite_training_step(
        "test", 3, 1f0, Inf32)
end

@testset "actionable GPU OOM diagnostics" begin
    error = try
        Reels._with_oom_diagnostics(
                "Wan", zeros(Float16, 4, 3, 2),
                TrainingConfig(activation_checkpointing=true,
                    checkpoint_interval=1),
                LowVRAMConfig()) do
            throw(Reels.CUDA.OutOfGPUMemoryError())
        end
        nothing
    catch caught
        caught
    end
    @test error isa ErrorException
    message = sprint(showerror, error)
    @test occursin("active latent bucket 4x3x2", message)
    @test occursin("dtype=Float16", message)
    @test occursin("activation_checkpointing=true", message)
    @test occursin("Int8 frozen weights", message)
end

@testset "mixed-precision AdamW master parameters" begin
    parameter = Float16.(Float32[1, -2])
    state = AdamWState([parameter])
    @test eltype(parameter) === Float16
    @test eltype(only(state.master)) === Float32
    @test only(state.master) == Float32[1, -2]
    update!(AdamW(learning_rate=1f-2, weight_decay=0f0), state,
            [parameter], [Float32[0.25, -0.5]])
    @test state.step == 1
    @test only(state.master) != Float32[1, -2]
    @test parameter == Float16.(only(state.master))

    mktempdir() do directory
        training = TrainingState(1, 3, Xoshiro(9), state)
        path = joinpath(directory, "mixed.reels")
        save_checkpoint(path, training, [parameter];
            metadata=Dict("model_family" => "synthetic",
                          "base_model" => "tiny"))
        restored = load_checkpoint(path)
        @test only(restored.state.optimizer.master) == only(state.master)
        @test only(restored.params) == Float32.(parameter)
        @test restored.metadata == Dict(
            "model_family" => "synthetic", "base_model" => "tiny")
    end
end

@testset "AdamW named weight-decay exclusions" begin
    names = ["blocks.0.attn.lora_A.weight", "blocks.0.attn.bias"]
    mask = weight_decay_mask(names, ["bias\$"])
    @test mask == [true, false]
    parameters = [ones(Float32, 2), ones(Float32, 2)]
    state = AdamWState(parameters)
    update!(AdamW(learning_rate=0.1f0, weight_decay=0.5f0), state,
        parameters, [zeros(Float32, 2), zeros(Float32, 2)];
        weight_decay_mask=mask)
    @test parameters[1] == fill(0.95f0, 2)
    @test parameters[2] == ones(Float32, 2)
    @test_throws DimensionMismatch update!(
        AdamW(), state, parameters, [zeros(Float32, 2) for _ in 1:2];
        weight_decay_mask=[true])
end

@testset "static loss scaling" begin
    @test Reels._validated_loss_scale(1024) === 1024f0
    @test_throws ArgumentError Reels._validated_loss_scale(0)
    @test_throws ArgumentError Reels._validated_loss_scale(Inf)
    gradients = [Float16[1, -2]]
    unscaled = Reels._unscale_gradients(gradients, 128f0)
    @test eltype(only(unscaled)) === Float32
    @test only(unscaled) == Float32[1 / 128, -2 / 128]
    @test Reels._unscale_gradients(gradients, 1f0) === gradients
end

@testset "bit-exact BF16 flow sample" begin
    latents32 = reshape(Float32[-1, -0.25, 0.5, 1], 2, 2, 1)
    noise32 = reshape(Float32[0.5, 1, -1, 0.25], 2, 2, 1)
    latents = Reels.bfloat16_values(latents32)
    noise = Reels.bfloat16_values(noise32)
    sampled = flow_sample(latents, noise, BFloat16[0.25])
    @test eltype(sampled.noisy) === BFloat16
    @test Reels.float32_values(sampled.noisy) ≈
          0.75f0 .* Reels.float32_values(latents) .+
          0.25f0 .* Reels.float32_values(noise) atol=4f-3
    @test Reels.float32_values(sampled.target) ≈
          Reels.float32_values(noise) .-
          Reels.float32_values(latents) atol=4f-3
end

@testset "official LTX shifted logit-normal timesteps" begin
    @test ltx_timestep_shift(1024) ≈ 0.95f0
    @test ltx_timestep_shift(4096) ≈ 2.05f0
    first_rng = Xoshiro(92)
    second_rng = Xoshiro(92)
    first = shifted_logit_normal_timestep(
        first_rng, 10_000, 1024; uniform_probability=0)
    repeated = shifted_logit_normal_timestep(
        second_rng, 10_000, 1024; uniform_probability=0)
    @test first == repeated
    @test all(value -> 0f0 <= value <= 1f0, first)
    shifted = shifted_logit_normal_timestep(
        Xoshiro(92), 10_000, 4096; uniform_probability=0)
    @test sum(shifted) > sum(first)
    automatic = sample_flow_timesteps(
        FlowMatchingConfig(), :ltx23, Xoshiro(93), 8, 2048)
    uniform = sample_flow_timesteps(
        FlowMatchingConfig(), :wan21, Xoshiro(93), 8, 2048)
    @test automatic != uniform
end
