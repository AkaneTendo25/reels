using Test
using Random
using Reels
using CUDA

Sys.islinux() &&
    ccall(:prctl, Cint, (Cint, Cstring, Culong, Culong, Culong),
          15, "reels-unit-test", 0, 0, 0)

length(ARGS) == 1 ||
    error("usage: distributed_worker.jl OUTPUT_DIR")
output_dir = only(ARGS)
runtime = init_distributed(DistributedConfig(
    enabled=true, timeout_seconds=60.0))
rank = distributed_rank(runtime)

try
    @testset "NCCL all-reduce rank $rank" begin
        values = CUDA.fill(Float32(rank + 1), 4)
        allreduce_mean!(runtime, values)
        @test Array(values) == fill(1.5f0, 4)
    end

    @testset "distributed Wan training rank $rank" begin
        rng = MersenneTwister(810)
        config = Wan21Config(
            patch_size=(1, 2, 2), text_length=2,
            input_channels=1, hidden_size=6, ffn_size=8,
            frequency_size=4, text_size=5, output_channels=1,
            heads=1, layers=1)
        base = WanTransformer(config; rng=rng)
        base.head.weight .= randn(rng, Float32, size(base.head.weight))
        cpu = inject_wan_lora(base;
            targets=["blocks.0.self_attn.q.weight"],
            rank=1, alpha=1, rng=rng)
        model = move_to_device(cpu, :cuda, :fp32)
        transfer = array_transfer(:cuda, :fp32)
        latents = transfer(fill(Float32(rank + 1), 1, 1, 2, 4, 1))
        context = transfer(fill(Float32(2rank + 1), 5, 2, 1))
        noise = transfer(fill(Float32(rank) + 0.25f0, size(latents)))
        timesteps = transfer(Float32[0.4])
        batch = WanLatentBatch(
            latents, context; noise=noise, timesteps=timesteps)
        job = WanTrainingJob(
            model=model, batch=_ -> batch,
            training=TrainingConfig(
                steps=1, gradient_accumulation=1,
                learning_rate=0.01f0, weight_decay=0f0,
                max_gradient_norm=10f0, seed=811),
            output_dir=output_dir,
            checkpoint=CheckpointConfig(every_steps=1, keep_last=1),
            base_model="tiny-distributed-wan",
            distributed=runtime)

        result = train!(job)
        @test result.state.step == 1
        @test only(result.metrics).distributed_world_size == 2
        @test only(result.metrics).distributed_rank == rank
        if rank == 0
            @test result.adapter !== nothing
            @test result.metrics_path !== nothing
            @test result.summary !== nothing
        else
            @test result.adapter === nothing
            @test result.metrics_path === nothing
            @test result.summary === nothing
        end
        write_safetensors(
            joinpath(output_dir, "rank-$rank.safetensors"),
            wan_lora_state_dict(model);
            metadata=Dict("rank" => string(rank)))
        distributed_barrier!(runtime)
    end
finally
    close_distributed!(runtime)
end
