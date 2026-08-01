using Test
using Random
using Reels
using CUDA

length(ARGS) == 1 ||
    error("usage: distributed_check.jl OUTPUT_DIR")
output_dir = only(ARGS)

@testset "two-rank NCCL training artifacts" begin
    rank0 = open_tensor_source(joinpath(output_dir, "rank-0.safetensors"))
    rank1 = open_tensor_source(joinpath(output_dir, "rank-1.safetensors"))
    keys0 = sort!(tensor_keys(rank0))
    @test keys0 == sort!(tensor_keys(rank1))
    @test all(keys0) do key
        load_safetensor(rank0, key) == load_safetensor(rank1, key)
    end
    @test isfile(joinpath(output_dir, "adapter-final.safetensors"))
    @test isfile(joinpath(output_dir, "checkpoint-1.reels"))
    @test load_checkpoint(
        joinpath(output_dir, "checkpoint-1.reels")).state.step == 1
    metrics = readlines(joinpath(output_dir, "metrics.jsonl"))
    @test length(metrics) == 1
    record = Reels.parse_json(only(metrics))
    @test record["distributed_rank"] == 0
    @test record["distributed_world_size"] == 2

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
    latents = transfer(cat(
        fill(1f0, 1, 1, 2, 4, 1),
        fill(2f0, 1, 1, 2, 4, 1); dims=5))
    context = transfer(cat(
        fill(1f0, 5, 2, 1),
        fill(3f0, 5, 2, 1); dims=3))
    noise = transfer(cat(
        fill(0.25f0, 1, 1, 2, 4, 1),
        fill(1.25f0, 1, 1, 2, 4, 1); dims=5))
    timesteps = transfer(Float32[0.4, 0.4])
    batch = WanLatentBatch(
        latents, context; noise=noise, timesteps=timesteps)
    baseline = train!(WanTrainingJob(
        model=model, batch=_ -> batch,
        training=TrainingConfig(
            steps=1, gradient_accumulation=1,
            learning_rate=0.01f0, weight_decay=0f0,
            max_gradient_norm=10f0, seed=811),
        output_dir=joinpath(output_dir, "single-baseline"),
        checkpoint=CheckpointConfig(every_steps=1, keep_last=1),
        base_model="tiny-distributed-wan"))
    baseline_path = joinpath(
        output_dir, "single-baseline", "rank-format.safetensors")
    write_safetensors(baseline_path, wan_lora_state_dict(model))
    baseline_source = open_tensor_source(baseline_path)
    distributed_state = Dict(
        key => load_safetensor(rank0, key) for key in keys0)
    baseline_state = Dict(
        key => load_safetensor(baseline_source, key) for key in keys0)
    @test keys(distributed_state) == keys(baseline_state)
    @test all(key -> isapprox(
        distributed_state[key], baseline_state[key];
        rtol=2f-5, atol=2f-6), keys0)
    @test only(baseline.losses) ≈ record["loss"] rtol=2f-5 atol=2f-6
end
