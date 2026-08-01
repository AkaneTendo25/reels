using CUDA

Sys.islinux() &&
    ccall(:prctl, Cint, (Cint, Cstring, Culong, Culong, Culong),
          15, "reels-unit-test", 0, 0, 0)

@testset "CUDA Int8 frozen weights and CPU offload LoRA step" begin
    CUDA.functional() || error("CUDA test requested without a functional GPU")
    rng = Xoshiro(960)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
        text_size=5, output_channels=1, heads=1, layers=1)
    latents_host = randn(rng, Float32, 1, 1, 2, 4, 1)
    context_host = randn(rng, Float32, 5, 2, 1)
    noise_host = randn(rng, Float32, size(latents_host))
    timesteps_host = Float32[0.4]

    for cpu_offload in (false, true)
        model_rng = Xoshiro(961)
        base = WanTransformer(config; rng=model_rng)
        base.head.weight .= randn(
            model_rng, Float32, size(base.head.weight))
        adapted = inject_wan_lora(base;
            targets=["blocks.0.self_attn.q.weight"],
            rank=1, alpha=1, rng=model_rng)
        transfer = frozen_weight_transfer(
            :cuda, :fp16; quantization=:int8,
            cpu_offload=cpu_offload)
        model = move_to_device(adapted, transfer)
        frozen = model.blocks[1].self_attention.q.weight
        @test frozen isa QuantizedMatrix
        @test (frozen.values isa CUDA.CuArray) == !cpu_offload
        latents = Float16.(CUDA.CuArray(latents_host))
        context = Float16.(CUDA.CuArray(context_host))
        noise = Float16.(CUDA.CuArray(noise_host))
        timesteps = Float16.(CUDA.CuArray(timesteps_host))
        flow = flow_sample(latents, noise, timesteps)
        result = wan_lora_loss_and_gradients(
            model, flow.noisy, timesteps, context, flow.target;
            checkpoint_interval=1)
        @test isfinite(result.loss)
        @test length(result.gradients) == 2
        @test all(gradient -> all(isfinite, Array(gradient)),
                  result.gradients)
        CUDA.synchronize()
        CUDA.reclaim()
    end
end

@testset "CUDA LTX Int8 CPU offload keeps AdaLN on device" begin
    rng = Xoshiro(962)
    config = LTX23Config(video_only=true, video_heads=2, video_head_dim=4,
        video_channels=5, video_context_dim=6, layers=1,
        video_max_positions=(8, 16, 16))
    base = LTXVideoTransformer(config; rng=rng)
    adapted = inject_ltx23_lora(base;
        targets=["transformer_blocks.0.attn1.to_q.weight"],
        rank=2, alpha=2, rng=rng)
    transfer = frozen_weight_transfer(
        :cuda, :fp16; quantization=:int8, cpu_offload=true)
    model = move_to_device(adapted, transfer)
    array = array_transfer(:cuda, :fp16)
    latents = array(randn(rng, Float32, 5, 3, 1))
    context = array(randn(rng, Float32, 6, 2, 1))
    timesteps = array(Float32[0.4])
    target = array(randn(rng, Float32, 5, 3, 1))
    positions = CUDA.zeros(Float32, 3, 3, 1)
    parameters = [entry.value for entry in ltx23_lora_parameters(model)]
    result = ltx23_lora_step!(
        model, AdamW(learning_rate=1f-2, weight_decay=0f0),
        AdamWState(parameters), latents, timesteps, context, positions,
        target; checkpoint_interval=1)
    CUDA.synchronize()
    @test isfinite(result.loss)
    @test result.gradient_norm > 0f0
    @test all(gradient -> all(isfinite, Array(gradient)),
              Reels.ltx23_lora_loss_and_gradients(
                  model, latents, timesteps, context, positions, target;
                  checkpoint_interval=1).gradients)
    CUDA.reclaim()
end
