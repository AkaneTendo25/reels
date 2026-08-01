@testset "Wan LoRA reverse-mode gradient and optimizer step" begin
    rng = Xoshiro(61)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
        input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
        text_size=5, output_channels=1, heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    model = inject_wan_lora(base;
        targets=["blocks.0.self_attn.q.weight"], rank=1, alpha=1,
        dropout=0.25, train_bias=true, rng=rng)
    video = randn(rng, Float32, 1, 1, 2, 4, 1) # two patch tokens
    text = randn(rng, Float32, 5, 2, 1)
    timesteps = Float32[0.3]
    target = randn(rng, Float32, size(video))

    differentiated = wan_lora_loss_and_gradients(
        model, video, timesteps, text, target; dropout_seed=UInt64(9))
    @test isfinite(differentiated.loss)
    @test differentiated.names == [
        "blocks.0.self_attn.q.lora_A.weight",
        "blocks.0.self_attn.q.lora_B.weight",
        "blocks.0.self_attn.q.bias",
    ]
    @test length(differentiated.gradients) == 3
    @test all(gradient -> all(isfinite, gradient),
              differentiated.gradients)
    loss_scaled = wan_lora_loss_and_gradients(
        model, video, timesteps, text, target;
        loss_scale=128f0, dropout_seed=UInt64(9))
    @test loss_scaled.loss ≈ differentiated.loss rtol=2f-6 atol=2f-6
    @test all(eachindex(differentiated.gradients)) do index
        loss_scaled.gradients[index] ≈ differentiated.gradients[index]
    end
    @test_throws ArgumentError wan_lora_loss_and_gradients(
        model, video, timesteps, text, target; loss_scale=0f0)
    checkpointed = wan_lora_loss_and_gradients(
        model, video, timesteps, text, target;
        checkpoint_interval=1, dropout_seed=UInt64(9))
    @test checkpointed.loss ≈ differentiated.loss
    @test all(eachindex(differentiated.gradients)) do index
        checkpointed.gradients[index] ≈ differentiated.gradients[index]
    end

    layer = only(wan_lora_layers(model)).layer
    @test layer.train_bias
    @test layer.bias !== base.blocks[1].self_attention.q.bias
    @test any(!iszero, differentiated.gradients[3])
    analytic = differentiated.gradients[2][1, 1]
    epsilon = 2f-3
    original = layer.B[1, 1]
    layer.B[1, 1] = original + epsilon
    plus = sum(abs2, wan_transformer_forward(
        model, video, timesteps, text;
        training=true, dropout_seed=UInt64(9)) .- target) / length(target)
    layer.B[1, 1] = original - epsilon
    minus = sum(abs2, wan_transformer_forward(
        model, video, timesteps, text;
        training=true, dropout_seed=UInt64(9)) .- target) / length(target)
    layer.B[1, 1] = original
    finite_difference = (plus - minus) / (2f0 * epsilon)
    @test analytic ≈ finite_difference rtol=3f-2 atol=3f-3

    parameters = [entry.value for entry in wan_lora_parameters(model)]
    before = map(copy, parameters)
    state = AdamWState(parameters)
    step = wan_lora_step!(model, AdamW(learning_rate=1f-2,
        weight_decay=0f0), state, video, timesteps, text, target;
        dropout_seed=UInt64(9))
    @test step.state.step == 1
    @test step.gradient_norm > 0
    @test isfinite(step.loss)
    @test any(i -> parameters[i] != before[i], eachindex(parameters))
    comparison = wan_validation_comparison(
        model, randn(rng, Float32, size(video)), text; steps=2)
    @test comparison.mean_absolute_delta > 0
    @test comparison.enabled != comparison.disabled
    @test only(wan_lora_layers(model)).layer.enabled
end

@testset "Wan I2V LoRA training step" begin
    rng = Xoshiro(62)
    config = Wan21Config(
        variant=:i2v_test, model_type=:i2v,
        patch_size=(1, 2, 2), text_length=2,
        input_channels=36, hidden_size=6, ffn_size=8,
        frequency_size=4, text_size=5, output_channels=16,
        heads=1, layers=1)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    model = inject_wan_lora(base;
        targets=["blocks.0.cross_attn.v_img.weight"],
        rank=1, alpha=1, dropout=0.25, rng=rng)
    noisy = randn(rng, Float32, 16, 1, 2, 4, 1)
    conditioning = randn(rng, Float32, 20, 1, 2, 4, 1)
    text = randn(rng, Float32, 5, 2, 1)
    image = randn(rng, Float32, 1280, 3, 1)
    timesteps = Float32[0.25]
    target = randn(rng, Float32, size(noisy))
    parameters = [entry.value for entry in wan_lora_parameters(model)]
    before = map(copy, parameters)
    result = wan_lora_step!(
        model, AdamW(learning_rate=1f-2, weight_decay=0f0),
        AdamWState(parameters), noisy, timesteps, text, target;
        checkpoint_interval=1, dropout_seed=UInt64(10),
        conditioning_video=conditioning,
        image_features=image)
    @test isfinite(result.loss)
    @test result.gradient_norm > 0f0
    @test any(index -> parameters[index] != before[index],
              eachindex(parameters))
    comparison = wan_validation_comparison(
        model, copy(noisy), text; steps=2,
        conditioning_video=conditioning, image_features=image)
    @test comparison.mean_absolute_delta > 0f0
    batch = WanValidationBatch("I2V test", copy(noisy), text;
        conditioning_video=conditioning, image_features=image)
    @test batch.conditioning_video === conditioning
    @test batch.image_features === image
end
