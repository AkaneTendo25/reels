mutable struct TrainingState
    step::Int
    micro_step::Int
    rng::Xoshiro
    optimizer::AdamWState
end

const CHECKPOINT_PREFIX = UInt8[0x52, 0x45, 0x45, 0x4c, 0x53, 0x43, 0x4b]
const CHECKPOINT_VERSION = UInt8(3)
const REELS_VERSION = "0.1.0"

function _adapter_layout_signature(adapter_layers)
    join((
        "$(entry.path)|rank=$(size(entry.layer.A, 1))|" *
        "alpha=$(entry.layer.alpha)|dropout=$(entry.layer.dropout)|" *
        "train_bias=$(entry.layer.train_bias)|" *
        "A=$(join(size(entry.layer.A), 'x'))|" *
        "B=$(join(size(entry.layer.B), 'x'))"
        for entry in adapter_layers), ";")
end

_read_values(io, ::Type{T}, n::Integer) where T =
    read!(io, Vector{T}(undef, Int(n)))
function _write_array(io, a)
    host = Array(float32_values(a))
    write(io, Int32(ndims(host)))
    write(io, collect(Int64, size(host)))
    write(io, Int64(length(host)))
    write(io, vec(host))
end
function _read_array(io)
    nd = read(io, Int32)
    dims = Tuple(_read_values(io, Int64, nd))
    n = read(io, Int64)
    reshape(_read_values(io, Float32, n), dims)
end

function save_checkpoint(path::AbstractString, state::TrainingState, params;
                         metadata=Dict{String,String}())
    isempty(dirname(path)) || mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, CHECKPOINT_PREFIX); write(io, CHECKPOINT_VERSION)
        write(io, Int64(state.step)); write(io, Int64(state.micro_step))
        write(io, Int32(fieldcount(Xoshiro)))
        for i in 1:fieldcount(Xoshiro)
            fieldtype(Xoshiro, i) === UInt64 ||
                error("unsupported Xoshiro layout in this Julia version")
            write(io, getfield(state.rng, i))
        end
        write(io, Int32(length(params)))
        foreach(p -> _write_array(io, p), params)
        write(io, Int64(state.optimizer.step))
        foreach(a -> _write_array(io, a), state.optimizer.m)
        foreach(a -> _write_array(io, a), state.optimizer.v)
        foreach(a -> _write_array(io, a), state.optimizer.master)
        encoded = json_encode(Dict{String,String}(
            String(key) => String(value) for (key, value) in metadata))
        write(io, Int64(ncodeunits(encoded)))
        write(io, codeunits(encoded))
    end
    mv(tmp, path; force=true)
    path
end

function load_checkpoint(path::AbstractString)
    open(path, "r") do io
        _read_values(io, UInt8, length(CHECKPOINT_PREFIX)) == CHECKPOINT_PREFIX ||
            throw(ArgumentError("not a Reels checkpoint"))
        version = read(io, UInt8)
        version in (UInt8(1), UInt8(2), CHECKPOINT_VERSION) ||
            throw(ArgumentError("unsupported Reels checkpoint version $version"))
        step, micro = Int(read(io, Int64)), Int(read(io, Int64))
        rng_fields = Int(read(io, Int32))
        rng_fields == fieldcount(Xoshiro) ||
            throw(ArgumentError("checkpoint RNG layout is incompatible with this Julia version"))
        rng = Xoshiro(_read_values(io, UInt64, rng_fields)...)
        n = Int(read(io, Int32))
        params = [_read_array(io) for _ in 1:n]
        optstep = Int(read(io, Int64))
        m = [_read_array(io) for _ in 1:n]
        v = [_read_array(io) for _ in 1:n]
        master = version >= 2 ? [_read_array(io) for _ in 1:n] : map(copy, params)
        metadata = if version >= 3
            metadata_length = read(io, Int64)
            0 <= metadata_length <= 16 * 1024 * 1024 ||
                throw(ArgumentError("checkpoint metadata length is invalid"))
            parsed = parse_json(String(
                _read_values(io, UInt8, metadata_length)))
            parsed isa AbstractDict ||
                throw(ArgumentError("checkpoint metadata must be an object"))
            Dict{String,String}(String(key) => String(value)
                                for (key, value) in parsed)
        else
            Dict{String,String}()
        end
        (state=TrainingState(step, micro, rng,
                            AdamWState(optstep, m, v, master)), params=params,
         metadata=metadata)
    end
end

function _training_checkpoint_metadata(family::AbstractString,
                                       base_model::AbstractString,
                                       training::TrainingConfig,
                                       checkpoint::CheckpointConfig,
                                       validation::ValidationConfig,
                                       flow::FlowMatchingConfig,
                                       low_vram::LowVRAMConfig,
                                       adapter_layers,
                                       metric)
    ranks = unique(size(entry.layer.A, 1) for entry in adapter_layers)
    alphas = unique(entry.layer.alpha for entry in adapter_layers)
    Dict{String,String}(
        "reels_version" => REELS_VERSION,
        "julia_version" => string(VERSION),
        "model_family" => String(family),
        "base_model" => String(base_model),
        "cache_schema_version" => "1",
        "training.steps" => string(training.steps),
        "training.micro_batch_size" => string(training.micro_batch_size),
        "training.gradient_accumulation" =>
            string(training.gradient_accumulation),
        "training.learning_rate" => string(training.learning_rate),
        "training.weight_decay" => string(training.weight_decay),
        "training.weight_decay_exclusions" =>
            json_encode(training.weight_decay_exclusions),
        "training.max_gradient_norm" =>
            string(training.max_gradient_norm),
        "training.loss_scale" => string(training.loss_scale),
        "training.seed" => string(training.seed),
        "training.activation_checkpointing" =>
            string(training.activation_checkpointing),
        "training.checkpoint_interval" =>
            string(training.checkpoint_interval),
        "training.scheduler" => string(training.scheduler.kind),
        "training.warmup_steps" =>
            string(training.scheduler.warmup_steps),
        "checkpoint.every_steps" => string(checkpoint.every_steps),
        "checkpoint.keep_last" => string(checkpoint.keep_last),
        "validation.every_steps" => string(validation.every_steps),
        "validation.inference_steps" =>
            string(validation.inference_steps),
        "validation.prompts" => json_encode(validation.prompts),
        "flow_matching.timestep_sampling" =>
            string(flow.timestep_sampling),
        "flow_matching.standard_deviation" =>
            string(flow.standard_deviation),
        "flow_matching.epsilon" => string(flow.epsilon),
        "flow_matching.uniform_probability" =>
            string(flow.uniform_probability),
        "low_vram.frozen_weight_quantization" =>
            string(low_vram.frozen_weight_quantization),
        "low_vram.cpu_offload" => string(low_vram.cpu_offload),
        "adapter.rank" =>
            length(ranks) == 1 ? string(only(ranks)) : "mixed",
        "adapter.alpha" =>
            length(alphas) == 1 ? string(only(alphas)) : "mixed",
        "adapter.train_bias" =>
            string(any(entry -> entry.layer.train_bias, adapter_layers)),
        "adapter.layout" => _adapter_layout_signature(adapter_layers),
        "metrics.step" => string(metric.step),
        "metrics.loss" => string(metric.loss),
        "metrics.smoothed_loss" => string(metric.smoothed_loss),
        "metrics.learning_rate" => string(metric.learning_rate),
        "metrics.gradient_norm" => string(metric.gradient_norm),
        "distributed.world_size" =>
            string(metric.distributed_world_size),
    )
end

function _resume_smoothed_loss(metadata::AbstractDict)
    saved = get(metadata, "metrics.smoothed_loss", "")
    isempty(saved) && return nothing
    value = tryparse(Float32, saved)
    value === nothing || !isfinite(value) ?
        throw(ArgumentError(
            "checkpoint smoothed-loss metric is invalid")) : value
end

function _validate_resume_value(metadata::AbstractDict, key::String,
                                current)
    saved = get(metadata, key, "")
    isempty(saved) && return
    value = string(current)
    saved == value ||
        throw(ArgumentError(
            "resume configuration mismatch for $key: checkpoint=$saved, " *
            "current=$value"))
end

function _validate_resume_configuration(metadata::AbstractDict,
                                        training::TrainingConfig,
                                        flow::FlowMatchingConfig,
                                        adapter_layers)
    isempty(metadata) && return
    for (key, value) in (
        "training.micro_batch_size" => training.micro_batch_size,
        "training.gradient_accumulation" =>
            training.gradient_accumulation,
        "training.learning_rate" => training.learning_rate,
        "training.weight_decay" => training.weight_decay,
        "training.weight_decay_exclusions" =>
            json_encode(training.weight_decay_exclusions),
        "training.max_gradient_norm" => training.max_gradient_norm,
        "training.loss_scale" => training.loss_scale,
        "training.activation_checkpointing" =>
            training.activation_checkpointing,
        "training.checkpoint_interval" => training.checkpoint_interval,
        "training.scheduler" => training.scheduler.kind,
        "training.warmup_steps" => training.scheduler.warmup_steps,
        "flow_matching.timestep_sampling" => flow.timestep_sampling,
        "flow_matching.standard_deviation" => flow.standard_deviation,
        "flow_matching.epsilon" => flow.epsilon,
        "flow_matching.uniform_probability" =>
            flow.uniform_probability,
        "adapter.layout" => _adapter_layout_signature(adapter_layers),
    )
        _validate_resume_value(metadata, key, value)
    end
    # Linear and cosine schedules depend on the configured final step.
    training.scheduler.kind === :constant ||
        _validate_resume_value(
            metadata, "training.steps", training.steps)
    nothing
end

function _validate_resume_identity(metadata::AbstractDict,
                                   family::AbstractString,
                                   base_model::AbstractString)
    isempty(metadata) && return
    get(metadata, "model_family", "") == family ||
        throw(ArgumentError("checkpoint model family does not match the job"))
    saved_base = get(metadata, "base_model", "")
    isempty(saved_base) || saved_base == base_model ||
        throw(ArgumentError("checkpoint base model does not match the job"))
end

function _validate_resume_world_size(metadata::AbstractDict,
                                     world_size::Integer)
    isempty(metadata) && return
    saved_value = get(metadata, "distributed.world_size", "")
    isempty(saved_value) && return
    saved_world_size = tryparse(Int, saved_value)
    saved_world_size === nothing &&
        throw(ArgumentError(
            "checkpoint distributed world size is invalid"))
    saved_world_size == world_size ||
        throw(ArgumentError(
            "checkpoint was created with distributed world size " *
            "$saved_world_size, but this job uses $world_size"))
    nothing
end

function _validate_resume_low_vram(metadata::AbstractDict,
                                   config::LowVRAMConfig)
    isempty(metadata) && return
    saved_quantization = get(
        metadata, "low_vram.frozen_weight_quantization", "")
    isempty(saved_quantization) ||
        saved_quantization == string(config.frozen_weight_quantization) ||
        throw(ArgumentError(
            "checkpoint frozen-weight quantization does not match the job"))
    saved_offload = get(metadata, "low_vram.cpu_offload", "")
    isempty(saved_offload) ||
        saved_offload == string(config.cpu_offload) ||
        throw(ArgumentError(
            "checkpoint CPU-offload setting does not match the job"))
    nothing
end
