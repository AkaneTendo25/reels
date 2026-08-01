using Test
using Random
using Reels
using CUDA

@testset "CUDA 100-step Wan overfit and memory stability" begin
    CUDA.device!(0)
    rng = Xoshiro(901)
    config = Wan21Config(
        patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=8, ffn_size=16,
        frequency_size=4, text_size=6, output_channels=1,
        heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    adapted = inject_wan_lora(base;
        targets=lora_targets(Wan21(), base, :attention_and_ffn),
        rank=4, alpha=4, rng=rng)
    model = move_to_device(adapted, :cuda, :bf16)
    transfer = array_transfer(:cuda, :bf16)
    latents = transfer(randn(rng, Float32, 1, 1, 2, 4, 1))
    context = transfer(randn(rng, Float32, 6, 2, 1))
    noise = transfer(randn(rng, Float32, size(latents)))
    timesteps = transfer(Float32[0.35])
    batch = WanLatentBatch(
        latents, context; noise=noise, timesteps=timesteps)
    training = TrainingConfig(
        steps=100, gradient_accumulation=1,
        learning_rate=0.03f0, weight_decay=0f0,
        max_gradient_norm=10f0, seed=902,
        activation_checkpointing=true, checkpoint_interval=1)

    mktempdir() do directory
        job = WanTrainingJob(
            model=model, batch=_ -> batch, training=training,
            output_dir=directory,
            checkpoint=CheckpointConfig(every_steps=10, keep_last=1),
            base_model="tiny-cuda-overfit")
        started = time_ns()
        result = train!(job)
        CUDA.synchronize()
        elapsed = Float64(time_ns() - started) / 1_000_000_000

        @test result.state.step == 100
        @test all(isfinite, result.losses)
        @test minimum(result.losses[end-9:end]) <
              0.7f0 * result.losses[1]
        @test all(metric -> metric.gpu_memory_bytes > 0,
                  result.metrics)
        @test all(metric -> metric.gpu_pool_reserved_bytes >=
                            metric.gpu_pool_used_bytes >= 0,
                  result.metrics)
        stable_memory = getproperty.(result.metrics[end-19:end],
                                     :gpu_memory_bytes)
        @test maximum(stable_memory) - minimum(stable_memory) <= 64 * 1024^2
        @test isfile(result.metrics_path)
        @test isfile(result.summary)
        records = Reels.parse_json.(readlines(result.metrics_path))
        @test length(records) == 100
        @test records[end]["gpu_peak_memory_bytes"] >=
              records[end]["gpu_memory_bytes"]
        validation_noise = transfer(randn(rng, Float32, size(latents)))
        comparison = wan_validation_comparison(
            model, validation_noise, context; steps=2)
        @test isfinite(comparison.mean_absolute_delta)
        @test comparison.mean_absolute_delta > 0f0
        println((
            initial_loss=result.losses[1],
            final_loss=result.losses[end],
            best_final_window=minimum(result.losses[end-9:end]),
            validation_mean_absolute_delta=
                comparison.mean_absolute_delta,
            elapsed_seconds=elapsed,
            optimizer_steps_per_second=100 / elapsed,
            peak_gpu_memory_bytes=maximum(
                getproperty.(result.metrics, :gpu_peak_memory_bytes)),
            stable_memory_span_bytes=
                maximum(stable_memory) - minimum(stable_memory),
        ))
    end
end
