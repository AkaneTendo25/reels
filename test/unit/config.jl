@testset "configuration" begin
    mktempdir() do dir
        path = joinpath(dir, "config.toml")
        write(path, """
        schema_version = 1
        seed = 7
        output_dir = "run"
        [model]
        family = "synthetic"
        variant = "tiny"
        checkpoint = "none"
        precision = "fp32"
        [data]
        manifest = "data.toml"
        cache_dir = "cache"
        frame_buckets = [9]
        resolution_buckets = ["64x32"]
        [adapter]
        rank = 2
        train_bias = true
        [adapter.rank_overrides]
        "blocks.0.self_attn.q" = 1
        [adapter.alpha_overrides]
        "blocks.0.self_attn.q.weight" = 3.0
        [training]
        steps = 3
        learning_rate = 0.01
        weight_decay_exclusions = ["bias\$"]
        [validation]
        every_steps = 2
        prompts = ["A test clip."]
        [flow_matching]
        timestep_sampling = "shifted_logit_normal"
        uniform_probability = 0.2
        [distributed]
        enabled = true
        world_size = 2
        rank = 1
        local_rank = 1
        master_port = 29501
        [low_vram]
        frozen_weight_quantization = "int8"
        cpu_offload = false
        """)
        cfg = load_config_file(path)
        @test cfg.model.family == :synthetic
        @test cfg.adapter.train_bias
        @test cfg.adapter.rank_overrides["blocks.0.self_attn.q"] == 1
        @test cfg.adapter.alpha_overrides[
            "blocks.0.self_attn.q.weight"] == 3f0
        @test cfg.training.weight_decay_exclusions == ["bias\$"]
        @test cfg.data.resolution_buckets == [(64, 32)]
        @test cfg.training.activation_checkpointing
        @test cfg.training.checkpoint_interval == 1
        @test cfg.validation.every_steps == 2
        @test cfg.validation.prompts == ["A test clip."]
        @test cfg.flow_matching.timestep_sampling == :shifted_logit_normal
        @test cfg.flow_matching.uniform_probability == 0.2f0
        @test cfg.distributed.enabled
        @test cfg.distributed.world_size == 2
        @test cfg.distributed.rank == cfg.distributed.local_rank == 1
        @test cfg.distributed.master_port == 29501
        @test cfg.low_vram.frozen_weight_quantization == :int8
        @test !cfg.low_vram.cpu_offload
        @test isempty(validate(cfg))
        write(path, replace(read(path, String), "rank = 2" => "rank = 2\nbogus = 1"))
        @test_throws ArgumentError load_config_file(path)
        write(path, replace(read(path, String), "bogus = 1" => "rank_only = 1"))
        @test_throws ArgumentError load_config_file(path)
    end
end

@testset "FP16 static loss-scale configuration" begin
    configuration(precision, loss_scale) = ReelsConfig(
        output_dir="run",
        model=ModelConfig(family=:synthetic, variant="tiny",
            checkpoint="none", precision=precision),
        data=DataConfig(manifest="data.toml", cache_dir="cache"),
        training=TrainingConfig(loss_scale=loss_scale))

    fp16 = configuration(:fp16, 1024f0)
    @test isempty(validate(fp16))
    @test fp16.training.loss_scale == 1024f0
    @test any(error -> occursin("model.precision=fp16", error),
        validate(configuration(:fp32, 1024f0)))
    @test any(error -> occursin("finite and positive", error),
        validate(configuration(:fp16, 0f0)))
end

@testset "Wan I2V configuration requires image encoder" begin
    mktempdir() do directory
        path = joinpath(directory, "i2v.toml")
        write(path, """
        output_dir = "run"
        [model]
        family = "wan21"
        variant = "i2v-14b-480p"
        checkpoint = "wan"
        image_encoder_checkpoint = "clip"
        [data]
        manifest = "data.toml"
        cache_dir = "cache"
        """)
        config = load_config_file(path)
        @test config.model.image_encoder_checkpoint == "clip"
        @test isempty(validate(config))
        write(path, replace(read(path, String),
            "image_encoder_checkpoint = \"clip\"\n" => ""))
        @test_throws ArgumentError load_config_file(path)
    end
end
