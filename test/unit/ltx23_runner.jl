@testset "LTX-2.3 inference schedules and RES-2S" begin
    distilled = ltx23_sigma_schedule(8, 1024; kind=:distilled)
    @test distilled == Reels.LTX23_DISTILLED_SIGMAS
    @test_throws ArgumentError ltx23_sigma_schedule(7, 1024; kind=:distilled)
    @test_throws ArgumentError ltx23_sigma_schedule(8, 1024; kind=:unknown)

    shifted = ltx23_sigma_schedule(15, 1280)
    @test length(shifted) == 16
    @test first(shifted) == 1f0
    @test last(shifted) == 0f0
    @test all(diff(shifted) .< 0f0)
    @test shifted != flow_euler_timesteps(15)

    sample = reshape(Float32[1, 2], 2, 1, 1)
    x0 = reshape(Float32[0.25, -0.5], 2, 1, 1)
    @test Reels._ltx23_res2s_midpoint(sample, x0, 0.5f0, 0f0) === nothing
    @test Reels._ltx23_res2s_step(sample, x0, zero(x0), 0.5f0, 0f0) == x0
end

@testset "LTX-2.3 cache provider and resumable training loop" begin
    rng = Xoshiro(232)
    config = LTX23Config(video_only=true, video_heads=1, video_head_dim=6,
        video_channels=3, video_context_dim=4, layers=1,
        video_max_positions=(8, 16, 16))
    latents = randn(rng, Float32, 3, 2)
    context = randn(rng, Float32, 4, 2)
    positions = zeros(Float32, 3, 2)

    mktempdir() do dir
        entries = String[]
        for index in 1:2
            path = joinpath(dir, "cache-$index.safetensors")
            write_ltx23_preprocess_cache(path, "key-$index",
                latents .+ index, context .+ index, positions)
            @test inspect_ltx23_preprocess_cache(path).valid
            push!(entries, path)
        end
        provider = LTXCachedBatchProvider(entries; batch_size=2)
        batch = provider(Xoshiro(1))
        @test batch isa LTXLatentBatch
        @test size(batch.latents) == (3, 2, 2)
        @test size(batch.text_context) == (4, 2, 2)
        @test size(batch.positions) == (3, 2, 2)
    end

    noise = randn(rng, Float32, 3, 2, 1)
    fixed = LTXLatentBatch(reshape(latents, 3, 2, 1),
        reshape(context, 4, 2, 1), reshape(positions, 3, 2, 1);
        noise=noise, timesteps=Float32[0.35])

    function makejob(directory, steps)
        model_rng = Xoshiro(233)
        base = LTXVideoTransformer(config; rng=model_rng)
        model = inject_ltx23_lora(base;
            targets=["transformer_blocks.0.attn1.to_q.weight"],
            rank=1, alpha=1, dropout=0.25, train_bias=true,
            rng=model_rng)
        training = TrainingConfig(steps=steps, gradient_accumulation=1,
            learning_rate=0.01f0, weight_decay=0f0,
            weight_decay_exclusions=["bias\$"],
            max_gradient_norm=10f0, seed=234)
        LTXTrainingJob(model=model, batch=_ -> fixed, training=training,
            output_dir=directory,
            checkpoint=CheckpointConfig(every_steps=1, keep_last=1),
            base_model="tiny-ltx",
            validation=ValidationConfig(every_steps=1,
                prompts=["tiny LTX validation"], inference_steps=2),
            validation_batches=[LTXValidationBatch(
                "tiny LTX validation", noise,
                reshape(context, 4, 2, 1),
                reshape(positions, 3, 2, 1))])
    end

    mktempdir() do dir
        full = makejob(joinpath(dir, "full"), 2)
        initial_lora = deepcopy(ltx23_lora_state_dict(full.model))
        full_result = train!(full)
        @test full_result.state.step == 2
        @test full_result.state.micro_step == 2
        @test all(isfinite, full_result.losses)
        @test ltx23_lora_state_dict(full.model) != initial_lora
        @test isfile(full_result.adapter)
        @test isfile(full_result.metrics_path)
        @test isfile(full_result.summary)
        metric_records = Reels.parse_json.(readlines(full_result.metrics_path))
        @test getindex.(metric_records, "step") == [1, 2]
        @test all(record -> record["model_family"] == "ltx23",
                  metric_records)
        @test all(record -> isfinite(record["smoothed_loss"]),
                  metric_records)
        @test all(record -> record["active_latent_bucket"] == "3x2",
                  metric_records)
        @test all(record -> record["target_rms"] > 0 &&
                            record["noisy_rms"] > 0 &&
                            isapprox(record["error_rms"], sqrt(record["loss"]);
                                     rtol=1e-6),
                  metric_records)
        summary = Reels.parse_json(read(full_result.summary, String))
        @test summary["validation_count"] == 2
        @test summary["adapter"] == "adapter-final.safetensors"
        @test length(full_result.validations) == 2
        @test all(event -> isfile(event.path), full_result.validations)
        @test all(event -> event.mean_absolute_delta > 0,
                  full_result.validations)
        @test !isfile(joinpath(dir, "full", "checkpoint-1.reels"))
        @test isfile(joinpath(dir, "full", "checkpoint-2.reels"))
        checkpoint_metadata = load_checkpoint(
            joinpath(dir, "full", "checkpoint-2.reels")).metadata
        @test checkpoint_metadata["model_family"] == "ltx23"
        @test checkpoint_metadata["base_model"] == "tiny-ltx"
        @test checkpoint_metadata["metrics.step"] == "2"
        @test checkpoint_metadata["training.loss_scale"] == "1.0"
        @test checkpoint_metadata["adapter.train_bias"] == "true"

        partial = makejob(joinpath(dir, "partial"), 1)
        train!(partial)
        resumed = makejob(joinpath(dir, "resumed"), 2)
        resumed_result = train!(resumed;
            resume_from=joinpath(dir, "partial", "checkpoint-1.reels"))
        @test resumed_result.state.step == 2
        @test ltx23_lora_state_dict(resumed.model) ==
            ltx23_lora_state_dict(full.model)
        @test resumed_result.state.optimizer.m ==
            full_result.state.optimizer.m
        @test resumed_result.state.optimizer.v ==
            full_result.state.optimizer.v
        resumed_metrics =
            Reels.parse_json.(readlines(resumed_result.metrics_path))
        @test getindex.(resumed_metrics, "step") == [2]
        @test only(getindex.(resumed_metrics, "smoothed_loss")) ==
            metric_records[2]["smoothed_loss"]
    end
end
