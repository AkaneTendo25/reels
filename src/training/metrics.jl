const METRICS_FORMAT = "reels-training-metrics-v1"
const RUN_SUMMARY_FORMAT = "reels-run-summary-v1"

_elapsed_seconds(start_ns::Integer) =
    Float64(time_ns() - start_ns) / 1_000_000_000

function _synchronize_training_device(array)
    array isa CUDA.CuArray && CUDA.synchronize()
    nothing
end

function _adapter_parameter_norm(parameters)
    norm_squared = 0f0
    for parameter in parameters
        norm_squared += Float32(sum(abs2, float32_values(parameter)))
    end
    Float32(sqrt(norm_squared))
end

function _assert_finite_training_step(family::AbstractString,
                                      step::Integer,
                                      loss::Real,
                                      gradient_norm::Real)
    isfinite(loss) ||
        throw(DomainError(loss,
            "$family produced a non-finite loss before optimizer step $step"))
    isfinite(gradient_norm) ||
        throw(DomainError(gradient_norm,
            "$family produced non-finite LoRA gradients before optimizer " *
            "step $step; no update was applied"))
    nothing
end

function _with_oom_diagnostics(
        operation::Function, family::AbstractString, latents,
        training::TrainingConfig, low_vram::LowVRAMConfig)
    try
        operation()
    catch error
        error isa CUDA.OutOfGPUMemoryError || rethrow()
        bucket = join(size(latents), 'x')
        throw(ErrorException(
            "$family GPU out of memory in active latent bucket $bucket " *
            "with dtype=$(eltype(latents)), " *
            "activation_checkpointing=$(training.activation_checkpointing), " *
            "checkpoint_interval=$(training.checkpoint_interval), " *
            "frozen_weight_quantization=" *
            "$(low_vram.frozen_weight_quantization), " *
            "cpu_offload=$(low_vram.cpu_offload). Reduce the frame or " *
            "resolution bucket / micro-batch size, keep activation " *
            "checkpointing enabled with interval 1, or enable Int8 frozen " *
            "weights and CPU offload."))
    end
end

function _training_memory_metrics(array)
    array isa CUDA.CuArray || return (
        gpu_memory_bytes=Int64(0),
        gpu_pool_used_bytes=Int64(0),
        gpu_pool_reserved_bytes=Int64(0),
    )
    pool_used_bytes = CUDA.used_memory()
    pool_reserved_bytes = CUDA.cached_memory()
    used_bytes = Int64(coalesce(pool_used_bytes, 0))
    reserved_bytes = Int64(coalesce(pool_reserved_bytes, 0))
    (
        # Driver-level free memory can become a wrapped negative value when a
        # stream-ordered pool temporarily overcommits physical memory. The pool
        # counters are process-local and are the quantities training can
        # meaningfully attribute and compare across steps.
        gpu_memory_bytes=used_bytes,
        gpu_pool_used_bytes=used_bytes,
        gpu_pool_reserved_bytes=reserved_bytes,
    )
end

function _reclaim_training_memory(array)
    array isa CUDA.CuArray || return nothing
    # Reverse-mode graphs contain many short-lived CuArrays whose finalizers
    # need not run before the next optimizer step. Force their release at the
    # natural step boundary, then return unused pool pages to CUDA so long
    # runs have bounded live and reserved memory.
    GC.gc(true)
    CUDA.reclaim()
    nothing
end

function _metric_dictionary(metric)
    Dict{String,Any}(String(name) => getfield(metric, name)
                     for name in propertynames(metric))
end

function _prepare_metrics_log(output_dir::AbstractString, resume_step::Integer)
    mkpath(output_dir)
    path = joinpath(output_dir, "metrics.jsonl")
    retained = String[]
    if resume_step > 0 && isfile(path)
        for line in eachline(path)
            isempty(strip(line)) && continue
            parsed = parse_json(line)
            parsed isa AbstractDict ||
                throw(ArgumentError("invalid training metric in $path"))
            step = get(parsed, "step", -1)
            step isa Integer ||
                throw(ArgumentError("training metric step is not an integer"))
            step <= resume_step && push!(retained, line)
        end
    end
    tmp = path * ".tmp"
    open(tmp, "w") do io
        foreach(line -> println(io, line), retained)
    end
    mv(tmp, path; force=true)
    path
end

function _append_training_metric(path::AbstractString, family::AbstractString,
                                 metric)
    record = _metric_dictionary(metric)
    record["format"] = METRICS_FORMAT
    record["model_family"] = String(family)
    record["recorded_at"] = string(now(UTC))
    open(path, "a") do io
        println(io, json_encode(record))
    end
    path
end

function _write_run_summary(output_dir::AbstractString, family::AbstractString,
                            state, metrics, validations, adapter)
    latest = isempty(metrics) ? Dict{String,Any}() :
        _metric_dictionary(last(metrics))
    summary = Dict{String,Any}(
        "format" => RUN_SUMMARY_FORMAT,
        "model_family" => String(family),
        "status" => "completed",
        "completed_at" => string(now(UTC)),
        "step" => state.step,
        "micro_step" => state.micro_step,
        "metric_count" => length(metrics),
        "validation_count" => length(validations),
        "adapter" => basename(adapter),
        "metrics" => "metrics.jsonl",
        "latest_metric" => latest,
    )
    path = joinpath(output_dir, "run-summary.json")
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, json_encode(summary))
        write(io, '\n')
    end
    mv(tmp, path; force=true)
    path
end
function _tensor_rms(array)
    values = float32_values(array)
    Float32(sqrt(Float32(sum(abs2, values)) / Float32(length(values))))
end
