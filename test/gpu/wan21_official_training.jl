using Test
using Random
using Reels
using CUDA

checkpoint = get(ENV, "WAN21_1_3B_CHECKPOINT", "")
isempty(checkpoint) &&
    error("set WAN21_1_3B_CHECKPOINT to the official Wan 2.1 1.3B SafeTensors file")

@testset "official Wan 2.1 1.3B BF16 LoRA overfit" begin
    backend = Wan21(variant=:t2v_1_3b, checkpoint=checkpoint)
    base = load_transformer(backend, checkpoint, :cpu, :fp32)
    adapted = inject_wan_lora(base;
        targets=lora_targets(backend, base, :attention),
        rank=1, alpha=1, rng=Xoshiro(1201))
    model = move_to_device(adapted, :cuda, :bf16)
    base = nothing
    adapted = nothing
    GC.gc(true)
    CUDA.reclaim()

    transfer = array_transfer(:cuda, :bf16)
    rng = Xoshiro(1202)
    latents = transfer(randn(rng, Float32, 16, 1, 2, 2, 1))
    context = transfer(randn(rng, Float32, 4096, 1, 1))
    noise = transfer(randn(rng, Float32, size(latents)))
    timesteps = transfer(Float32[0.4])
    batch = WanLatentBatch(
        latents, context; noise=noise, timesteps=timesteps)
    trainable = wan_lora_parameters(model)
    before = Array(trainable[2].value)
    training = TrainingConfig(
        steps=3, gradient_accumulation=1,
        learning_rate=1f-3, weight_decay=0f0,
        max_gradient_norm=1f0, seed=1203,
        activation_checkpointing=true, checkpoint_interval=1)

    mktempdir() do directory
        job = WanTrainingJob(
            model=model, batch=_ -> batch, training=training,
            output_dir=directory,
            checkpoint=CheckpointConfig(every_steps=1, keep_last=1),
            base_model=checkpoint)
        result = train!(job)
        CUDA.synchronize()

        @test result.state.step == 3
        @test length(result.losses) == 3
        @test all(isfinite, result.losses)
        @test result.losses[end] < result.losses[1]
        @test all(metric -> metric.gradient_norm > 0f0, result.metrics)
        @test all(metric -> metric.gpu_memory_bytes > 0, result.metrics)
        memory = getproperty.(result.metrics, :gpu_memory_bytes)
        @test maximum(memory) - minimum(memory) <= 512 * 1024^2
        @test Array(wan_lora_parameters(model)[2].value) != before
        @test isfile(result.adapter)
        @test isfile(result.metrics_path)
        @test isfile(result.summary)
        header = inspect_safetensors(result.adapter)
        @test get(header.metadata, "format", "") == Reels.WAN_LORA_FORMAT
        println((
            initial_loss=result.losses[1],
            final_loss=result.losses[end],
            gradient_norm=result.metrics[end].gradient_norm,
            gpu_memory_bytes=result.metrics[end].gpu_memory_bytes,
            gpu_pool_reserved_bytes=
                result.metrics[end].gpu_pool_reserved_bytes,
            gpu_peak_memory_bytes=result.metrics[end].gpu_peak_memory_bytes,
            gpu_memory_span_bytes=maximum(memory) - minimum(memory),
            adapter_tensors=length(header.tensors),
        ))
    end
end
