function _cli_options(arguments)
    options = Dict{String,String}()
    positionals = String[]
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if startswith(argument, "--")
            index < length(arguments) ||
                throw(ArgumentError("missing value for $argument"))
            options[argument[3:end]] = arguments[index + 1]
            index += 2
        else
            push!(positionals, argument)
            index += 1
        end
    end
    positionals, options
end

function _wan_variant(value::AbstractString)
    normalized = replace(lowercase(value), '-' => '_', '.' => '_')
    normalized in ("t2v_1_3b", "wan21_t2v_1_3b") && return :t2v_1_3b
    normalized in ("t2v_14b", "wan21_t2v_14b") && return :t2v_14b
    normalized in ("i2v_14b", "i2v_14b_480p", "wan21_i2v_14b",
                   "wan21_i2v_14b_480p") && return :i2v_14b_480p
    normalized in ("i2v_14b_720p", "wan21_i2v_14b_720p") &&
        return :i2v_14b_720p
    throw(ArgumentError("unsupported Wan variant $value"))
end

function _require_model_path(value::String, name::String)
    isempty(value) && throw(ArgumentError("$name must be set in [model]"))
    isfile(value) || throw(ArgumentError("$name does not exist: $value"))
    value
end

function _require_model_source(value::String, name::String)
    isempty(value) && throw(ArgumentError("$name must be set in [model]"))
    (isfile(value) || isdir(value)) ||
        throw(ArgumentError("$name does not exist: $value"))
    value
end

function _model_source_fingerprint(path::AbstractString)
    isfile(path) && return checkpoint_fingerprint(path)
    isdir(path) || throw(ArgumentError("model source does not exist: $path"))
    records = Any[]
    for (root, _, files) in walkdir(path), file in sort(files)
        full = joinpath(root, file)
        info = stat(full)
        push!(records, Dict(
            "path" => relpath(full, path),
            "size" => Int(info.size),
            "mtime" => string(info.mtime)))
    end
    bytes2hex(sha256(json_encode(records)))
end

function _cache_entries(directory::AbstractString)
    isdir(directory) || throw(ArgumentError("cache directory does not exist: $directory"))
    entries = String[]
    for (root, _, files) in walkdir(directory), file in files
        endswith(file, ".safetensors") || continue
        path = joinpath(root, file)
        try
            inspect_preprocess_cache(path).valid && push!(entries, path)
        catch
        end
    end
    sort!(entries)
end

function _ltx_cache_entries(directory::AbstractString)
    isdir(directory) ||
        throw(ArgumentError("cache directory does not exist: $directory"))
    entries = String[]
    for (root, _, files) in walkdir(directory), file in files
        endswith(file, ".safetensors") || continue
        path = joinpath(root, file)
        try
            inspect_ltx23_preprocess_cache(path).valid &&
                push!(entries, path)
        catch
        end
    end
    sort!(entries)
end

function _validation_batches(directory::AbstractString,
                             family::AbstractString,
                             prompts::AbstractVector{<:AbstractString};
                             device=:cpu, precision=:fp32)
    isempty(prompts) && return Any[]
    root = joinpath(directory, "validation")
    isdir(root) || throw(ArgumentError(
        "validation cache is missing; rerun `reels preprocess`"))
    candidates = Dict(prompt => String[] for prompt in prompts)
    for (path, _, files) in walkdir(root), file in files
        endswith(file, ".safetensors") || continue
        full = joinpath(path, file)
        inspected = try
            inspect_validation_cache(full)
        catch
            continue
        end
        inspected.valid || continue
        metadata = inspected.header.metadata
        get(metadata, "model_family", "") == family || continue
        prompt = get(metadata, "prompt", "")
        haskey(candidates, prompt) && push!(candidates[prompt], full)
    end
    loaded = map(prompts) do prompt
        paths = candidates[prompt]
        isempty(paths) && throw(ArgumentError(
            "no validation cache exists for prompt: $prompt"))
        length(paths) == 1 || throw(ArgumentError(
            "multiple validation caches exist for prompt `$prompt`; " *
            "remove stale entries and rerun preprocessing"))
        load_validation_cache(only(paths);
            device=device, precision=precision)
    end
    if family == "wan21"
        all(batch -> batch isa WanValidationBatch, loaded) ||
            throw(ArgumentError("Wan validation cache family mismatch"))
        return WanValidationBatch[batch for batch in loaded]
    elseif family == "ltx23"
        all(batch -> batch isa LTXValidationBatch, loaded) ||
            throw(ArgumentError("LTX validation cache family mismatch"))
        return LTXValidationBatch[batch for batch in loaded]
    end
    throw(ArgumentError("unsupported validation cache family $family"))
end

function cli_preprocess(config_path::AbstractString; device=:cuda)
    config = load_config_file(config_path)
    if config.model.family === :ltx23
        backend = LTX23(
            variant=Symbol(replace(config.model.variant, '-' => '_')),
            checkpoint=config.model.checkpoint, video_only=true)
        text_checkpoint = _require_model_source(
            config.model.text_encoder_checkpoint,
            "model.text_encoder_checkpoint")
        tokenizer_path = _require_model_path(config.model.tokenizer_model,
            "model.tokenizer_model")
        vae_checkpoint = _require_model_path(config.model.vae_checkpoint,
            "model.vae_checkpoint")
        connector_checkpoint = _require_model_path(config.model.checkpoint,
            "model.checkpoint")
        conditioner = load_text_encoder(backend, text_checkpoint;
            tokenizer_model=tokenizer_path,
            connector_checkpoint=connector_checkpoint,
            device=device, precision=config.model.precision)
        vae = load_vae(backend, vae_checkpoint, :cpu, :fp32)
        vae = move_to_device(vae, device, config.model.precision)
        samples = load_video_manifest(config.data.manifest)
        identity = PreprocessIdentity(
            model_family="ltx23",
            model_checkpoint=checkpoint_fingerprint(connector_checkpoint),
            text_encoder_checkpoint=_model_source_fingerprint(text_checkpoint),
            vae_checkpoint=checkpoint_fingerprint(vae_checkpoint),
            vae_scale=bytes2hex(sha256(json_encode(Dict(
                "scale_factors" => LTX23_VIDEO_SCALE_FACTORS,
                "normalization" => "native-vae-mean")))),
            dtype=string(config.model.precision),
        )
        created = 0
        entries = String[]
        try
            for (index, sample) in enumerate(samples)
                result = build_ltx23_preprocess_cache(
                    config.data.cache_dir, backend, conditioner, vae,
                    sample, identity, config.data.resolution_buckets,
                    config.data.frame_buckets;
                    target_fps=config.data.target_fps)
                created += result.created
                push!(entries, result.path)
                println("[$index/$(length(samples))] ",
                    result.created ? "created " : "cached  ", result.path)
            end
            validation = isempty(config.validation.prompts) ? NamedTuple[] :
                build_ltx23_validation_caches(
                    config.data.cache_dir, backend, conditioner, first(entries),
                    config.validation.prompts, identity; seed=config.seed)
            foreach(validation) do result
                println(result.created ? "created " : "cached  ", result.path)
            end
        finally
            close(conditioner)
        end
        return (samples=length(samples), created=created,
                reused=length(samples) - created,
                validation=length(config.validation.prompts))
    end
    config.model.family === :wan21 ||
        throw(ArgumentError("preprocess supports model.family=wan21 or ltx23"))
    backend = Wan21(variant=_wan_variant(config.model.variant),
        checkpoint=config.model.checkpoint)
    text_checkpoint = _require_model_path(config.model.text_encoder_checkpoint,
        "model.text_encoder_checkpoint")
    tokenizer_path = _require_model_path(config.model.tokenizer_model,
        "model.tokenizer_model")
    vae_checkpoint = _require_model_path(config.model.vae_checkpoint,
        "model.vae_checkpoint")
    image_encoder = if load_config(backend).model_type === :i2v
        image_checkpoint = _require_model_source(
            config.model.image_encoder_checkpoint,
            "model.image_encoder_checkpoint")
        load_image_encoder(backend, image_checkpoint;
            device=device, precision=config.model.precision)
    else
        nothing
    end
    conditioner = load_text_encoder(backend, text_checkpoint;
        tokenizer_model=tokenizer_path, device=device,
        precision=config.model.precision,
        quantization=config.low_vram.frozen_weight_quantization,
        cpu_offload=config.low_vram.cpu_offload)
    vae = load_wan_vae_encoder(vae_checkpoint)
    device === :cuda && (vae = move_to_device(vae, :cuda))
    samples = load_video_manifest(config.data.manifest)
    identity = PreprocessIdentity(
        model_family="wan21",
        model_checkpoint=isfile(config.model.checkpoint) ?
            checkpoint_fingerprint(config.model.checkpoint) :
            config.model.checkpoint,
        text_encoder_checkpoint=checkpoint_fingerprint(text_checkpoint),
        image_encoder_checkpoint=image_encoder === nothing ? "" :
            _model_source_fingerprint(
                config.model.image_encoder_checkpoint),
        vae_checkpoint=checkpoint_fingerprint(vae_checkpoint),
        vae_scale=bytes2hex(sha256(json_encode(Dict(
            "mean" => WAN_VAE_MEAN, "std" => WAN_VAE_STD)))),
        dtype=string(config.model.precision),
    )
    created = 0
    entries = String[]
    for (index, sample) in enumerate(samples)
        result = build_wan_preprocess_cache(config.data.cache_dir,
            backend, conditioner, vae, sample, identity,
            config.data.resolution_buckets, config.data.frame_buckets;
            target_fps=config.data.target_fps,
            image_encoder=image_encoder)
        created += result.created
        push!(entries, result.path)
        println("[$index/$(length(samples))] ",
            result.created ? "created " : "cached  ", result.path)
    end
    validation = isempty(config.validation.prompts) ? NamedTuple[] :
        build_wan_validation_caches(
            config.data.cache_dir, backend, conditioner, first(entries),
            config.validation.prompts, identity; seed=config.seed)
    foreach(validation) do result
        println(result.created ? "created " : "cached  ", result.path)
    end
    close(conditioner.tokenizer)
    (samples=length(samples), created=created,
     reused=length(samples) - created,
     validation=length(config.validation.prompts))
end

function cli_train(config_path::AbstractString; device=:cuda,
                   resume_from=nothing)
    config = load_config_file(config_path)
    config.distributed.enabled && device !== :cuda &&
        throw(ArgumentError("distributed training requires device=:cuda"))
    config.low_vram.cpu_offload && device !== :cuda &&
        throw(ArgumentError("CPU weight offloading requires device=:cuda"))
    runtime = init_distributed(config.distributed)
    callback = (_, metric) -> println(
        "step=$(metric.step) loss=$(metric.loss) lr=$(metric.learning_rate) " *
        "grad_norm=$(metric.gradient_norm)")
    try
        rank = distributed_rank(runtime)
        world_size = distributed_world_size(runtime)
        if config.model.family === :ltx23
            backend = LTX23(
                variant=Symbol(replace(config.model.variant, '-' => '_')),
                checkpoint=config.model.checkpoint, video_only=true)
            base = load_transformer(
                backend, config.model.checkpoint, :cpu, :fp32)
            model = inject_ltx23_lora(base;
                targets=lora_targets(backend, base, config.adapter.targets),
                rank=config.adapter.rank, alpha=config.adapter.alpha,
                dropout=config.adapter.dropout,
                train_bias=config.adapter.train_bias,
                rank_overrides=config.adapter.rank_overrides,
                alpha_overrides=config.adapter.alpha_overrides,
                rng=Xoshiro(config.seed))
            transfer = frozen_weight_transfer(
                device, config.model.precision;
                quantization=config.low_vram.frozen_weight_quantization,
                cpu_offload=config.low_vram.cpu_offload)
            model = move_to_device(model, transfer)
            provider = LTXCachedBatchProvider(
                _ltx_cache_entries(config.data.cache_dir);
                batch_size=config.training.micro_batch_size, device=device,
                precision=config.model.precision, rank=rank,
                world_size=world_size)
            validation_batches = is_main_process(runtime) ?
                _validation_batches(
                    config.data.cache_dir, "ltx23",
                    config.validation.prompts; device=device,
                    precision=config.model.precision) :
                LTXValidationBatch[]
            job = LTXTrainingJob(model=model, batch=provider,
                training=config.training, output_dir=config.output_dir,
                checkpoint=config.checkpoint,
                base_model=config.model.checkpoint,
                flow_matching=config.flow_matching,
                validation=config.validation,
                validation_batches=validation_batches,
                distributed=runtime, low_vram=config.low_vram)
            return train!(
                job; resume_from=resume_from, callback=callback)
        end
        config.model.family === :wan21 ||
            throw(ArgumentError(
                "train supports model.family=wan21 or ltx23"))
        backend = Wan21(
            variant=_wan_variant(config.model.variant),
            checkpoint=config.model.checkpoint)
        # Adapter initialization is CPU-side and the adapted model is
        # transferred once, keeping every LoRA matrix on the selected device.
        base = load_transformer(
            backend, config.model.checkpoint, :cpu, :fp32)
        model = inject_wan_lora(base;
            targets=lora_targets(backend, base, config.adapter.targets),
            rank=config.adapter.rank, alpha=config.adapter.alpha,
            dropout=config.adapter.dropout,
            train_bias=config.adapter.train_bias,
            rank_overrides=config.adapter.rank_overrides,
            alpha_overrides=config.adapter.alpha_overrides,
            rng=Xoshiro(config.seed))
        transfer = frozen_weight_transfer(
            device, config.model.precision;
            quantization=config.low_vram.frozen_weight_quantization,
            cpu_offload=config.low_vram.cpu_offload)
        model = move_to_device(model, transfer)
        entries = _cache_entries(config.data.cache_dir)
        provider = CachedBatchProvider(entries;
            batch_size=config.training.micro_batch_size, device=device,
            precision=config.model.precision, rank=rank,
            world_size=world_size)
        validation_batches = is_main_process(runtime) ?
            _validation_batches(
                config.data.cache_dir, "wan21",
                config.validation.prompts; device=device,
                precision=config.model.precision) :
            WanValidationBatch[]
        job = WanTrainingJob(model=model, batch=provider,
            training=config.training, output_dir=config.output_dir,
            checkpoint=config.checkpoint, base_model=config.model.checkpoint,
            flow_matching=config.flow_matching,
            validation=config.validation,
            validation_batches=validation_batches,
            distributed=runtime, low_vram=config.low_vram)
        train!(job; resume_from=resume_from, callback=callback)
    finally
        close_distributed!(runtime)
    end
end

function cli_inspect(path::AbstractString)
    isfile(path) || throw(ArgumentError("file does not exist: $path"))
    if endswith(lowercase(path), ".safetensors")
        header = inspect_safetensors(path)
        println("SafeTensors: ", path)
        println("metadata: ", header.metadata)
        println("tensors: ", length(header.tensors))
        for key in sort!(collect(keys(header.tensors)))
            tensor = header.tensors[key]
            println("  ", key, " ", tensor.dtype, " ", Tuple(tensor.shape))
        end
        return header
    end
    restored = load_checkpoint(path)
    println("training checkpoint: step=$(restored.state.step) ",
        "micro_step=$(restored.state.micro_step) params=$(length(restored.params))")
    isempty(restored.metadata) || println("metadata: ", restored.metadata)
    restored
end

function cli_validate_run(directory::AbstractString)
    isdir(directory) || throw(ArgumentError("run directory does not exist: $directory"))
    adapter = joinpath(directory, "adapter-final.safetensors")
    isfile(adapter) || throw(ArgumentError("missing final adapter: $adapter"))
    header = inspect_safetensors(adapter)
    get(header.metadata, "format", "") in
        (WAN_LORA_FORMAT, WAN_LORA_FORMAT_V1,
         LTX23_LORA_FORMAT, LTX23_LORA_FORMAT_V1) ||
        throw(ArgumentError("final adapter has an unsupported format"))
    checkpoints = _checkpoint_steps(directory)
    family = get(header.metadata, "model_family", "")
    for step in checkpoints
        path = joinpath(directory, "checkpoint-$step.reels")
        restored = load_checkpoint(path)
        restored.state.step == step ||
            throw(ArgumentError(
                "checkpoint filename and embedded step differ: $path"))
        isempty(restored.metadata) ||
            get(restored.metadata, "model_family", "") == family ||
            throw(ArgumentError(
                "checkpoint model family differs from final adapter: $path"))
    end
    validation_artifacts = String[]
    validation_root = joinpath(directory, "validation")
    if isdir(validation_root)
        for (root, _, files) in walkdir(validation_root), file in files
            endswith(file, ".safetensors") || continue
            path = joinpath(root, file)
            artifact = inspect_safetensors(path)
            get(artifact.metadata, "format", "") ==
                "reels-validation-latents-v1" ||
                throw(ArgumentError(
                    "unsupported validation artifact: $path"))
            sort!(collect(keys(artifact.tensors))) ==
                ["adapter_disabled", "adapter_enabled"] ||
                throw(ArgumentError(
                    "invalid validation tensor inventory: $path"))
            push!(validation_artifacts, path)
        end
    end
    metric_records = Dict{String,Any}[]
    metrics_path = joinpath(directory, "metrics.jsonl")
    if isfile(metrics_path)
        for line in eachline(metrics_path)
            isempty(strip(line)) && continue
            record = parse_json(line)
            record isa Dict{String,Any} ||
                throw(ArgumentError("invalid JSONL metric in $metrics_path"))
            get(record, "format", "") == METRICS_FORMAT ||
                throw(ArgumentError("unsupported metric format in $metrics_path"))
            get(record, "model_family", "") == family ||
                throw(ArgumentError("metric model family differs from adapter"))
            push!(metric_records, record)
        end
        metric_steps = [record["step"] for record in metric_records]
        all(step -> step isa Integer, metric_steps) ||
            throw(ArgumentError("metric steps must be integers"))
        all(>(0), diff(metric_steps)) ||
            throw(ArgumentError("metric steps must be strictly increasing"))
    end
    summary_path = joinpath(directory, "run-summary.json")
    summary_record = if isfile(summary_path)
        parsed = parse_json(read(summary_path, String))
        parsed isa Dict{String,Any} ||
            throw(ArgumentError("invalid run summary: $summary_path"))
        get(parsed, "format", "") == RUN_SUMMARY_FORMAT ||
            throw(ArgumentError("unsupported run summary format"))
        get(parsed, "model_family", "") == family ||
            throw(ArgumentError("run summary model family differs from adapter"))
        get(parsed, "status", "") == "completed" ||
            throw(ArgumentError("run summary does not report completion"))
        parsed
    else
        nothing
    end
    println("valid run: adapter tensors=$(length(header.tensors)) checkpoints=",
        checkpoints, " validation_artifacts=$(length(validation_artifacts))",
        " metrics=$(length(metric_records))")
    (adapter=adapter, checkpoints=checkpoints,
     validation_artifacts=sort!(validation_artifacts),
     metrics=metrics_path, metric_records=metric_records,
     summary=summary_path, summary_record=summary_record)
end

function reels_main(arguments=ARGS)
    isempty(arguments) && throw(ArgumentError(
        "usage: reels <preprocess|train|validate|inspect> [options]"))
    command = first(arguments)
    positionals, options = _cli_options(arguments[2:end])
    device = Symbol(get(options, "device", "cuda"))
    device in (:cpu, :cuda) || throw(ArgumentError("--device must be cpu or cuda"))
    if command == "preprocess"
        config = get(options, "config", "")
        isempty(config) && throw(ArgumentError("preprocess requires --config PATH"))
        return cli_preprocess(config; device=device)
    elseif command == "train"
        config = get(options, "config", "")
        isempty(config) && throw(ArgumentError("train requires --config PATH"))
        return cli_train(config; device=device,
            resume_from=get(options, "resume", nothing))
    elseif command == "inspect"
        length(positionals) == 1 ||
            throw(ArgumentError("inspect requires exactly one file path"))
        return cli_inspect(only(positionals))
    elseif command == "validate"
        directory = get(options, "run", "")
        isempty(directory) && throw(ArgumentError("validate requires --run PATH"))
        return cli_validate_run(directory)
    end
    throw(ArgumentError("unknown command $command"))
end
