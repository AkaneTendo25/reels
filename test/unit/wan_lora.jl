@testset "Wan LoRA injection and adapter round-trip" begin
    rng = Xoshiro(51)
    config = Wan21Config(patch_size=(1, 2, 2), text_length=5,
        input_channels=2, hidden_size=12, ffn_size=24, frequency_size=8,
        text_size=10, output_channels=2, heads=3, layers=2)
    base = WanTransformer(config; rng=rng)
    base.head.weight .= randn(rng, Float32, size(base.head.weight))
    targets = lora_targets(Wan21(), base, :attention)
    adapted = inject_wan_lora(base; targets=targets, rank=3, alpha=6,
                              rng=Xoshiro(52))

    layers = wan_lora_layers(adapted)
    parameters = wan_lora_parameters(adapted)
    @test length(layers) == 16
    @test length(parameters) == 32
    @test length(unique(entry.path for entry in layers)) == length(layers)
    @test first(layers).path == "blocks.0.self_attn.q"
    @test layers[1].layer.weight === base.blocks[1].self_attention.q.weight
    @test all(parameter -> parameter.value isa Matrix{Float32}, parameters)

    video = randn(rng, Float32, 2, 2, 4, 4, 1)
    text = randn(rng, Float32, 10, 5, 1)
    timesteps = Float32[0.4]
    expected_base = wan_transformer_forward(base, video, timesteps, text)
    @test wan_transformer_forward(adapted, video, timesteps, text) == expected_base

    for entry in layers
        entry.layer.B .= randn(rng, Float32, size(entry.layer.B)) .* 0.01f0
    end
    expected = wan_transformer_forward(adapted, video, timesteps, text)
    @test expected != expected_base

    mktempdir() do dir
        path = joinpath(dir, "adapter.safetensors")
        save_wan_lora(path, adapted; base_model="Wan2.1-T2V-1.3B",
                      metadata=Dict("test" => "roundtrip"))
        header = inspect_safetensors(path)
        @test header.metadata["format"] == "reels-wan21-lora-v2"
        @test header.metadata["scaling_baked_into_lora_B"] == "true"
        @test header.metadata["external_format"] ==
            "diffusers-non-diffusers-wan"
        @test header.metadata["base_model"] == "Wan2.1-T2V-1.3B"
        @test header.metadata["rank"] == "3"
        @test length(header.tensors) == 32
        @test haskey(header.tensors,
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight")
        source = open_tensor_source(path)
        exported_b = Reels.load_state_tensor(source, TensorSpec(
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight",
            collect(size(first(layers).layer.B)), LINEAR_OUT_IN))
        @test exported_b ≈ first(layers).layer.B .* 2f0

        restored = inject_wan_lora(base; targets=targets, rank=3, alpha=6,
                                   rng=Xoshiro(53))
        load_wan_lora!(restored, path)
        @test wan_lora_state_dict(restored) == wan_lora_state_dict(adapted)
        @test wan_transformer_forward(restored, video, timesteps, text) == expected
    end

    @test_throws ArgumentError inject_wan_lora(base;
        targets=[r"does_not_exist"], rank=2)

    biased = inject_wan_lora(base;
        targets=["blocks.0.self_attn.q.weight"], rank=4, alpha=4,
        train_bias=true,
        rank_overrides=Dict("blocks.0.self_attn.q" => 1),
        alpha_overrides=Dict("blocks.0.self_attn.q.weight" => 3f0),
        rng=Xoshiro(56))
    biased_layer = only(wan_lora_layers(biased)).layer
    @test size(biased_layer.A, 1) == 1
    @test biased_layer.alpha == 3f0
    @test biased_layer.train_bias
    @test biased_layer.bias !== base.blocks[1].self_attention.q.bias
    @test length(wan_lora_parameters(biased)) == 3
    biased_layer.bias .+= 0.25f0
    biased_output = wan_transformer_forward(
        biased, video, timesteps, text)
    @test biased_output != expected_base
    set_adapter_enabled!(biased_layer, false)
    @test wan_transformer_forward(
        biased, video, timesteps, text) == expected_base
    set_adapter_enabled!(biased_layer, true)
    mktempdir() do dir
        path = joinpath(dir, "biased-adapter.safetensors")
        save_wan_lora(path, biased)
        @test inspect_safetensors(path).metadata["train_bias"] == "true"
        restored = inject_wan_lora(base;
            targets=["blocks.0.self_attn.q.weight"], rank=1, alpha=3,
            train_bias=true, rng=Xoshiro(57))
        load_wan_lora!(restored, path)
        @test wan_lora_state_dict(restored) == wan_lora_state_dict(biased)
    end
    @test_throws ArgumentError inject_wan_lora(base;
        targets=["blocks.0.self_attn.q.weight"], rank=1,
        rank_overrides=Dict("blocks.9.self_attn.q" => 2))
end

@testset "Wan I2V adapter export round-trip" begin
    rng = Xoshiro(54)
    config = Wan21Config(variant=:i2v_test, model_type=:i2v,
        patch_size=(1, 2, 2), text_length=2, input_channels=36,
        hidden_size=8, ffn_size=12, frequency_size=4, text_size=6,
        output_channels=16, heads=2, layers=1)
    base = WanTransformer(config; rng=rng)
    adapted = inject_wan_lora(base;
        targets=lora_targets(Wan21(), base, :attention),
        rank=2, alpha=4, rng=rng)
    layers = wan_lora_layers(adapted)
    @test length(layers) == 10
    @test any(entry -> entry.path == "blocks.0.cross_attn.k_img", layers)
    @test any(entry -> entry.path == "blocks.0.cross_attn.v_img", layers)
    foreach(layers) do entry
        entry.layer.B .= randn(rng, Float32, size(entry.layer.B))
    end
    mktempdir() do directory
        path = joinpath(directory, "i2v-adapter.safetensors")
        save_wan_lora(path, adapted; base_model="Wan2.1-I2V-14B-480P")
        header = inspect_safetensors(path)
        @test length(header.tensors) == 20
        @test haskey(header.tensors,
            "diffusion_model.blocks.0.cross_attn.k_img.lora_A.weight")
        @test haskey(header.tensors,
            "diffusion_model.blocks.0.cross_attn.v_img.lora_B.weight")
        restored = inject_wan_lora(base;
            targets=lora_targets(Wan21(), base, :attention),
            rank=2, alpha=4, rng=Xoshiro(55))
        load_wan_lora!(restored, path)
        @test wan_lora_state_dict(restored) ==
            wan_lora_state_dict(adapted)
    end
end
