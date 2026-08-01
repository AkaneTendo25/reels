Base.@kwdef struct ModelConfig
    family::Symbol
    variant::String
    checkpoint::String
    text_encoder_checkpoint::String = ""
    image_encoder_checkpoint::String = ""
    tokenizer_model::String = ""
    vae_checkpoint::String = ""
    precision::Symbol = :bf16
end

Base.@kwdef struct DataConfig
    manifest::String
    cache_dir::String
    frame_buckets::Vector{Int} = [17, 33, 49]
    resolution_buckets::Vector{Tuple{Int,Int}} = [(512, 512)]
    target_fps::Int = 16
    workers::Int = 4
end

Base.@kwdef struct LoRAConfig
    rank::Int = 32
    alpha::Float32 = 32f0
    dropout::Float32 = 0f0
    targets::Symbol = :attention
    train_bias::Bool = false
    rank_overrides::Dict{String,Int} = Dict{String,Int}()
    alpha_overrides::Dict{String,Float32} = Dict{String,Float32}()
end

Base.@kwdef struct SchedulerConfig
    kind::Symbol = :constant
    warmup_steps::Int = 0
end

Base.@kwdef struct TrainingConfig
    steps::Int = 2_000
    micro_batch_size::Int = 1
    gradient_accumulation::Int = 1
    learning_rate::Float32 = 1f-4
    weight_decay::Float32 = 0.01f0
    weight_decay_exclusions::Vector{String} = String[]
    max_gradient_norm::Float32 = 1f0
    loss_scale::Float32 = 1f0
    seed::Int = 42
    activation_checkpointing::Bool = true
    checkpoint_interval::Int = 1
    scheduler::SchedulerConfig = SchedulerConfig()
end

Base.@kwdef struct CheckpointConfig
    every_steps::Int = 250
    keep_last::Int = 3
end

Base.@kwdef struct ValidationConfig
    every_steps::Int = 0
    prompts::Vector{String} = String[]
    inference_steps::Int = 20
end

Base.@kwdef struct FlowMatchingConfig
    timestep_sampling::Symbol = :auto
    standard_deviation::Float32 = 1f0
    epsilon::Float32 = 1f-3
    uniform_probability::Float32 = 0.1f0
end

Base.@kwdef struct DistributedConfig
    enabled::Bool = false
    backend::Symbol = :nccl
    world_size::Int = 0
    rank::Int = -1
    local_rank::Int = -1
    master_addr::String = "127.0.0.1"
    master_port::Int = 29_500
    timeout_seconds::Float64 = 120.0
end

Base.@kwdef struct LowVRAMConfig
    frozen_weight_quantization::Symbol = :none
    cpu_offload::Bool = false
end

Base.@kwdef struct ReelsConfig
    schema_version::Int = 1
    seed::Int = 42
    output_dir::String
    model::ModelConfig
    data::DataConfig
    adapter::LoRAConfig = LoRAConfig()
    training::TrainingConfig = TrainingConfig()
    checkpoint::CheckpointConfig = CheckpointConfig()
    validation::ValidationConfig = ValidationConfig()
    flow_matching::FlowMatchingConfig = FlowMatchingConfig()
    distributed::DistributedConfig = DistributedConfig()
    low_vram::LowVRAMConfig = LowVRAMConfig()
end

function validate(c::ReelsConfig)
    errors = String[]
    c.schema_version == 1 || push!(errors, "schema_version must be 1")
    c.model.family in (:wan21, :ltx23, :synthetic) ||
        push!(errors, "model.family must be wan21, ltx23, or synthetic")
    c.model.precision in (:bf16, :fp16, :fp32) ||
        push!(errors, "model.precision must be bf16, fp16, or fp32")
    if c.model.family in (:wan21, :ltx23)
        isempty(c.model.checkpoint) &&
            push!(errors, "model.checkpoint is required for $(c.model.family)")
    end
    if c.model.family === :wan21 &&
       occursin("i2v", lowercase(c.model.variant)) &&
       isempty(c.model.image_encoder_checkpoint)
        push!(errors,
            "model.image_encoder_checkpoint is required for Wan I2V")
    end
    c.adapter.rank > 0 || push!(errors, "adapter.rank must be positive")
    isfinite(c.adapter.alpha) && c.adapter.alpha > 0f0 ||
        push!(errors, "adapter.alpha must be finite and positive")
    0f0 <= c.adapter.dropout < 1f0 ||
        push!(errors, "adapter.dropout must be in [0, 1)")
    all(value -> value > 0, values(c.adapter.rank_overrides)) ||
        push!(errors, "adapter.rank_overrides values must be positive")
    all(value -> isfinite(value) && value > 0f0,
        values(c.adapter.alpha_overrides)) ||
        push!(errors,
            "adapter.alpha_overrides values must be finite and positive")
    c.training.scheduler.kind in (:constant, :linear, :cosine) ||
        push!(errors,
            "training.scheduler.type must be constant, linear, or cosine")
    c.training.scheduler.warmup_steps >= 0 ||
        push!(errors,
            "training.scheduler.warmup_steps cannot be negative")
    c.training.steps > 0 || push!(errors, "training.steps must be positive")
    c.training.micro_batch_size > 0 ||
        push!(errors, "training.micro_batch_size must be positive")
    c.training.gradient_accumulation > 0 ||
        push!(errors, "training.gradient_accumulation must be positive")
    c.training.learning_rate > 0 || push!(errors, "training.learning_rate must be positive")
    c.training.weight_decay >= 0 ||
        push!(errors, "training.weight_decay cannot be negative")
    foreach(c.training.weight_decay_exclusions) do pattern
        try
            Regex(pattern)
        catch
            push!(errors,
                "invalid training.weight_decay_exclusions regex: $pattern")
        end
    end
    c.training.max_gradient_norm > 0 ||
        push!(errors, "training.max_gradient_norm must be positive")
    isfinite(c.training.loss_scale) && c.training.loss_scale > 0f0 ||
        push!(errors, "training.loss_scale must be finite and positive")
    c.model.precision === :fp16 || c.training.loss_scale == 1f0 ||
        push!(errors,
            "training.loss_scale may differ from 1 only with model.precision=fp16")
    c.training.checkpoint_interval > 0 ||
        push!(errors, "training.checkpoint_interval must be positive")
    c.checkpoint.every_steps > 0 ||
        push!(errors, "checkpoint.every_steps must be positive")
    c.checkpoint.keep_last >= 0 ||
        push!(errors, "checkpoint.keep_last cannot be negative")
    c.validation.every_steps >= 0 ||
        push!(errors, "validation.every_steps cannot be negative")
    c.validation.inference_steps > 0 ||
        push!(errors, "validation.inference_steps must be positive")
    if c.validation.every_steps > 0 && isempty(c.validation.prompts)
        push!(errors, "validation.prompts cannot be empty when validation is enabled")
    end
    foreach(c.validation.prompts) do prompt
        isempty(strip(prompt)) &&
            push!(errors, "validation prompts cannot be empty")
    end
    c.flow_matching.timestep_sampling in
        (:auto, :uniform, :shifted_logit_normal) ||
        push!(errors, "flow_matching.timestep_sampling must be auto, uniform, or shifted_logit_normal")
    c.flow_matching.standard_deviation > 0 ||
        push!(errors, "flow_matching.standard_deviation must be positive")
    0 < c.flow_matching.epsilon < 0.5 ||
        push!(errors, "flow_matching.epsilon must be in (0, 0.5)")
    0 <= c.flow_matching.uniform_probability <= 1 ||
        push!(errors, "flow_matching.uniform_probability must be in [0, 1]")
    isempty(c.data.frame_buckets) && push!(errors, "data.frame_buckets cannot be empty")
    foreach(c.data.frame_buckets) do n
        n > 0 || push!(errors, "frame bucket $n must be positive")
    end
    c.distributed.backend === :nccl ||
        push!(errors, "distributed.backend must be nccl")
    c.distributed.world_size == 0 || c.distributed.world_size > 1 ||
        push!(errors, "distributed.world_size must be 0 or greater than 1")
    c.distributed.rank >= -1 ||
        push!(errors, "distributed.rank must be -1 or nonnegative")
    c.distributed.local_rank >= -1 ||
        push!(errors, "distributed.local_rank must be -1 or nonnegative")
    1 <= c.distributed.master_port <= 65_535 ||
        push!(errors, "distributed.master_port must be in 1:65535")
    c.distributed.timeout_seconds > 0 ||
        push!(errors, "distributed.timeout_seconds must be positive")
    c.low_vram.frozen_weight_quantization in (:none, :int8) ||
        push!(errors,
            "low_vram.frozen_weight_quantization must be none or int8")
    if (c.low_vram.frozen_weight_quantization !== :none ||
        c.low_vram.cpu_offload) && c.model.precision === :bf16
        push!(errors, "low-VRAM frozen weights require model.precision=fp16 or fp32")
    end
    errors
end
