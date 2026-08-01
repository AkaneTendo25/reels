const CONFIG_KEYS = Dict(
    "" => Set(["schema_version", "seed", "output_dir", "model", "data", "adapter",
               "training", "checkpoint", "validation", "flow_matching",
               "distributed", "low_vram"]),
    "model" => Set(["family", "variant", "checkpoint", "text_encoder_checkpoint",
                    "image_encoder_checkpoint", "tokenizer_model",
                    "vae_checkpoint", "precision"]),
    "data" => Set(["manifest", "cache_dir", "frame_buckets", "resolution_buckets",
                   "target_fps", "workers"]),
    "adapter" => Set(["type", "rank", "alpha", "dropout", "targets",
                      "train_bias", "rank_overrides", "alpha_overrides"]),
    "training" => Set(["steps", "micro_batch_size", "gradient_accumulation",
                       "learning_rate", "weight_decay", "max_gradient_norm", "seed",
                       "weight_decay_exclusions",
                       "loss_scale",
                       "activation_checkpointing", "checkpoint_interval",
                       "scheduler"]),
    "training.scheduler" => Set(["type", "warmup_steps"]),
    "checkpoint" => Set(["every_steps", "keep_last"]),
    "validation" => Set(["every_steps", "prompts", "inference_steps"]),
    "flow_matching" => Set(["timestep_sampling", "standard_deviation",
                            "epsilon", "uniform_probability"]),
    "distributed" => Set(["enabled", "backend", "world_size", "rank",
                          "local_rank", "master_addr", "master_port",
                          "timeout_seconds"]),
    "low_vram" => Set(["frozen_weight_quantization", "cpu_offload"]),
)

function _unknown_keys!(errors, table, path="")
    allowed = get(CONFIG_KEYS, path, nothing)
    allowed === nothing && return
    for key in keys(table)
        key in allowed || push!(errors, "unknown key: " * (isempty(path) ? key : "$path.$key"))
        value = table[key]
        value isa AbstractDict && _unknown_keys!(errors, value, isempty(path) ? key : "$path.$key")
    end
end

_resolution(s) = begin
    p = split(lowercase(s), 'x')
    length(p) == 2 || throw(ArgumentError("invalid resolution '$s'; expected WIDTHxHEIGHT"))
    (parse(Int, p[1]), parse(Int, p[2]))
end

function load_config_file(path::AbstractString; reject_unknown=true)
    raw = TOML.parsefile(path)
    errors = String[]
    reject_unknown && _unknown_keys!(errors, raw)
    isempty(errors) || throw(ArgumentError(join(errors, '\n')))
    m, d = raw["model"], raw["data"]
    a, t = get(raw, "adapter", Dict()), get(raw, "training", Dict())
    s, ck = get(t, "scheduler", Dict()), get(raw, "checkpoint", Dict())
    val = get(raw, "validation", Dict())
    flow = get(raw, "flow_matching", Dict())
    distributed = get(raw, "distributed", Dict())
    low_vram = get(raw, "low_vram", Dict())
    cfg = ReelsConfig(
        schema_version=get(raw, "schema_version", 1),
        seed=get(raw, "seed", 42),
        output_dir=raw["output_dir"],
        model=ModelConfig(family=Symbol(m["family"]), variant=m["variant"],
            checkpoint=m["checkpoint"],
            text_encoder_checkpoint=get(m, "text_encoder_checkpoint", ""),
            image_encoder_checkpoint=get(m, "image_encoder_checkpoint", ""),
            tokenizer_model=get(m, "tokenizer_model", ""),
            vae_checkpoint=get(m, "vae_checkpoint", ""),
            precision=Symbol(get(m, "precision", "bf16"))),
        data=DataConfig(manifest=d["manifest"], cache_dir=d["cache_dir"],
            frame_buckets=Int.(get(d, "frame_buckets", [17, 33, 49])),
            resolution_buckets=_resolution.(get(d, "resolution_buckets", ["512x512"])),
            target_fps=get(d, "target_fps", 16), workers=get(d, "workers", 4)),
        adapter=LoRAConfig(rank=get(a, "rank", 32), alpha=Float32(get(a, "alpha", 32)),
            dropout=Float32(get(a, "dropout", 0)), targets=Symbol(get(a, "targets", "attention")),
            train_bias=get(a, "train_bias", false),
            rank_overrides=Dict{String,Int}(
                String(key) => Int(value)
                for (key, value) in get(a, "rank_overrides", Dict())),
            alpha_overrides=Dict{String,Float32}(
                String(key) => Float32(value)
                for (key, value) in get(a, "alpha_overrides", Dict()))),
        training=TrainingConfig(steps=get(t, "steps", 2000),
            micro_batch_size=get(t, "micro_batch_size", 1),
            gradient_accumulation=get(t, "gradient_accumulation", 1),
            learning_rate=Float32(get(t, "learning_rate", 1e-4)),
            weight_decay=Float32(get(t, "weight_decay", 0.01)),
            weight_decay_exclusions=String.(
                get(t, "weight_decay_exclusions", String[])),
            max_gradient_norm=Float32(get(t, "max_gradient_norm", 1)),
            loss_scale=Float32(get(t, "loss_scale", 1)),
            seed=get(t, "seed", get(raw, "seed", 42)),
            activation_checkpointing=get(t, "activation_checkpointing", true),
            checkpoint_interval=get(t, "checkpoint_interval", 1),
            scheduler=SchedulerConfig(kind=Symbol(get(s, "type", "constant")),
                warmup_steps=get(s, "warmup_steps", 0))),
        checkpoint=CheckpointConfig(every_steps=get(ck, "every_steps", 250),
            keep_last=get(ck, "keep_last", 3)),
        validation=ValidationConfig(
            every_steps=get(val, "every_steps", 0),
            prompts=String.(get(val, "prompts", String[])),
            inference_steps=get(val, "inference_steps", 20)),
        flow_matching=FlowMatchingConfig(
            timestep_sampling=Symbol(get(flow, "timestep_sampling", "auto")),
            standard_deviation=Float32(get(flow, "standard_deviation", 1)),
            epsilon=Float32(get(flow, "epsilon", 1e-3)),
            uniform_probability=Float32(get(
                flow, "uniform_probability", 0.1))),
        distributed=DistributedConfig(
            enabled=get(distributed, "enabled", false),
            backend=Symbol(get(distributed, "backend", "nccl")),
            world_size=get(distributed, "world_size", 0),
            rank=get(distributed, "rank", -1),
            local_rank=get(distributed, "local_rank", -1),
            master_addr=String(get(
                distributed, "master_addr", "127.0.0.1")),
            master_port=get(distributed, "master_port", 29_500),
            timeout_seconds=Float64(get(
                distributed, "timeout_seconds", 120.0))),
        low_vram=LowVRAMConfig(
            frozen_weight_quantization=Symbol(get(
                low_vram, "frozen_weight_quantization", "none")),
            cpu_offload=get(low_vram, "cpu_offload", false)))
    errors = validate(cfg)
    isempty(errors) || throw(ArgumentError(join(errors, '\n')))
    cfg
end
