@testset "resumable Wan latent training loop" begin
    @test flow_euler_timesteps(4) ==
        Float32[1, 0.75, 0.5, 0.25, 0]
    shifted = flow_euler_timesteps(4; shift=5)
    @test shifted ≈ Float32[1, 15 / 16, 5 / 6, 5 / 8, 0]
    @test_throws ArgumentError flow_euler_timesteps(4; shift=0)

    rng = Xoshiro(71)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
        text_size=5, output_channels=1, heads=1, layers=1)
    latents = randn(rng, Float32, 1, 1, 2, 4, 1)
    context = randn(rng, Float32, 5, 2, 1)
    noise = randn(rng, Float32, size(latents))
    timesteps = Float32[0.4]
    batch = WanLatentBatch(latents, context; noise=noise, timesteps=timesteps)

    function makejob(output_dir, steps; gradient_accumulation=2)
        model_rng = Xoshiro(72)
        base = WanTransformer(config; rng=model_rng)
        base.head.weight .= randn(model_rng, Float32, size(base.head.weight))
        model = inject_wan_lora(base;
            targets=lora_targets(Wan21(), base, :attention),
            rank=1, alpha=1, dropout=0.25, train_bias=true,
            rng=model_rng)
        training = TrainingConfig(
            steps=steps, gradient_accumulation=gradient_accumulation,
            learning_rate=0.02f0, weight_decay=0f0,
            weight_decay_exclusions=["bias\$"],
            max_gradient_norm=10f0, seed=73)
        WanTrainingJob(model=model, batch=_ -> batch, training=training,
            output_dir=output_dir, checkpoint=CheckpointConfig(
                every_steps=2, keep_last=1), base_model="tiny-test",
            validation=ValidationConfig(every_steps=2,
                prompts=["tiny Wan validation"], inference_steps=2),
            validation_batches=[WanValidationBatch(
                "tiny Wan validation", noise, context)])
    end

    mktempdir() do dir
        callback_steps = Int[]
        full = makejob(joinpath(dir, "full"), 4)
        initial_lora = deepcopy(wan_lora_state_dict(full.model))
        full_result = train!(full;
            callback=(_, metric) -> push!(callback_steps, metric.step))
        @test full_result.state.step == 4
        @test full_result.state.micro_step == 8
        @test callback_steps == collect(1:4)
        @test all(isfinite, full_result.losses)
        @test wan_lora_state_dict(full.model) != initial_lora
        @test isfile(full_result.adapter)
        @test isfile(full_result.metrics_path)
        @test isfile(full_result.summary)
        metric_records = Reels.parse_json.(readlines(full_result.metrics_path))
        @test getindex.(metric_records, "step") == collect(1:4)
        @test all(record -> record["format"] == "reels-training-metrics-v1",
                  metric_records)
        @test all(record -> isfinite(record["smoothed_loss"]),
                  metric_records)
        @test all(record -> record["active_latent_bucket"] == "1x1x2x4",
                  metric_records)
        @test all(record -> record["target_rms"] > 0 &&
                            record["noisy_rms"] > 0 &&
                            isapprox(record["error_rms"], sqrt(record["loss"]);
                                     rtol=1e-6),
                  metric_records)
        @test all(record -> record["step_seconds"] >=
                            record["optimizer_seconds"] >= 0,
                  metric_records)
        summary = Reels.parse_json(read(full_result.summary, String))
        @test summary["format"] == "reels-run-summary-v1"
        @test summary["status"] == "completed"
        @test summary["step"] == 4
        @test length(full_result.validations) == 2
        @test all(event -> isfile(event.path), full_result.validations)
        @test all(event -> event.mean_absolute_delta > 0,
                  full_result.validations)
        @test !isfile(joinpath(dir, "full", "checkpoint-2.reels"))
        @test isfile(joinpath(dir, "full", "checkpoint-4.reels"))
        checkpoint_metadata = load_checkpoint(
            joinpath(dir, "full", "checkpoint-4.reels")).metadata
        @test checkpoint_metadata["model_family"] == "wan21"
        @test checkpoint_metadata["base_model"] == "tiny-test"
        @test checkpoint_metadata["metrics.step"] == "4"
        @test checkpoint_metadata["training.loss_scale"] == "1.0"
        @test checkpoint_metadata["adapter.train_bias"] == "true"
        @test checkpoint_metadata["training.weight_decay_exclusions"] ==
            "[\"bias\$\"]"
        @test checkpoint_metadata["validation.every_steps"] == "2"

        partial = makejob(joinpath(dir, "partial"), 2)
        train!(partial)
        checkpoint = joinpath(dir, "partial", "checkpoint-2.reels")
        incompatible = makejob(
            joinpath(dir, "incompatible"), 4;
            gradient_accumulation=1)
        mismatch = try
            train!(incompatible; resume_from=checkpoint)
            nothing
        catch error
            error
        end
        @test mismatch isa ArgumentError
        @test occursin(
            "training.gradient_accumulation", sprint(showerror, mismatch))
        resumed = makejob(joinpath(dir, "resumed"), 4)
        resumed_result = train!(resumed; resume_from=checkpoint)
        @test resumed_result.state.step == 4
        @test resumed_result.state.micro_step == 8
        @test wan_lora_state_dict(resumed.model) ==
            wan_lora_state_dict(full.model)
        @test resumed_result.state.optimizer.step ==
            full_result.state.optimizer.step
        @test resumed_result.state.optimizer.m ==
            full_result.state.optimizer.m
        @test resumed_result.state.optimizer.v ==
            full_result.state.optimizer.v
        resumed_metrics =
            Reels.parse_json.(readlines(resumed_result.metrics_path))
        @test getindex.(resumed_metrics, "step") == [3, 4]
        @test getindex.(resumed_metrics, "smoothed_loss") ==
            getindex.(metric_records[3:4], "smoothed_loss")
    end
end
