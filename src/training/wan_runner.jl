struct WanLatentBatch{L,C,N,T,V,I}
    latents::L
    text_context::C
    noise::N
    timesteps::T
    conditioning_video::V
    image_features::I
end
WanLatentBatch(latents, text_context; noise=nothing, timesteps=nothing,
               conditioning_video=nothing, image_features=nothing) =
    WanLatentBatch(latents, text_context, noise, timesteps,
                   conditioning_video, image_features)

Base.@kwdef struct WanTrainingJob{M,F,V,D}
    model::M
    batch::F
    training::TrainingConfig
    output_dir::String
    checkpoint::CheckpointConfig = CheckpointConfig()
    base_model::String = ""
    flow_matching::FlowMatchingConfig = FlowMatchingConfig()
    validation::ValidationConfig = ValidationConfig()
    validation_batches::V = WanValidationBatch[]
    distributed::D = SingleProcessRuntime()
    low_vram::LowVRAMConfig = LowVRAMConfig()
end

_device_random_normal(rng, like) =
    _constant_like(like, randn(rng, Float32, size(like)))
_device_timesteps(rng, like, batch) =
    _constant_like(like, uniform_timestep(rng, batch))
function _float32_state_like(like, saved)
    result = similar(like, Float32, size(saved))
    copyto!(result, float32_values(saved))
    result
end

function _restore_wan_state(restored, parameters)
    length(restored.params) == length(parameters) ||
        throw(DimensionMismatch("checkpoint parameter count differs"))
    for (parameter, saved) in zip(parameters, restored.params)
        size(parameter) == size(saved) ||
            throw(DimensionMismatch("checkpoint parameter shape differs"))
        copyto!(parameter, saved)
    end
    restored_state = restored.state
    length(restored_state.optimizer.m) == length(parameters) ||
        throw(DimensionMismatch("checkpoint optimizer state count differs"))
    m = [_float32_state_like(parameter, saved) for (parameter, saved) in
         zip(parameters, restored_state.optimizer.m)]
    v = [_float32_state_like(parameter, saved) for (parameter, saved) in
         zip(parameters, restored_state.optimizer.v)]
    masters = [_float32_state_like(parameter, saved) for (parameter, saved) in
               zip(parameters, restored_state.optimizer.master)]
    optimizer = AdamWState(restored_state.optimizer.step, m, v, masters)
    TrainingState(restored_state.step, restored_state.micro_step,
                  restored_state.rng, optimizer)
end
_restore_wan_state(path::AbstractString, parameters) =
    _restore_wan_state(load_checkpoint(path), parameters)

function _checkpoint_steps(output_dir)
    isdir(output_dir) || return Int[]
    steps = Int[]
    for name in readdir(output_dir)
        matched = match(r"^checkpoint-(\d+)\.reels$", name)
        matched === nothing || push!(steps, parse(Int, matched.captures[1]))
    end
    sort!(steps)
end

function _prune_wan_checkpoints!(job::WanTrainingJob)
    keep = job.checkpoint.keep_last
    keep > 0 || return
    steps = _checkpoint_steps(job.output_dir)
    for step in steps[1:max(0, length(steps) - keep)]
        checkpoint = joinpath(job.output_dir, "checkpoint-$step.reels")
        adapter = joinpath(job.output_dir, "adapter-$step.safetensors")
        isfile(checkpoint) && rm(checkpoint)
        isfile(adapter) && rm(adapter)
    end
end

function _save_wan_training_checkpoint(job, state, parameters, metric)
    mkpath(job.output_dir)
    step = state.step
    checkpoint = joinpath(job.output_dir, "checkpoint-$step.reels")
    adapter = joinpath(job.output_dir, "adapter-$step.safetensors")
    metadata = _training_checkpoint_metadata(
        "wan21", job.base_model, job.training, job.checkpoint,
        job.validation, job.flow_matching, job.low_vram,
        wan_lora_layers(job.model), metric)
    save_checkpoint(checkpoint, state, parameters; metadata=metadata)
    save_wan_lora(adapter, job.model; base_model=job.base_model,
        metadata=Dict("training_step" => string(step)))
    _prune_wan_checkpoints!(job)
    (checkpoint=checkpoint, adapter=adapter)
end

"""
    train!(job::WanTrainingJob; resume_from=nothing, callback=nothing)

Train injected Wan LoRA adapters from batches of cached video latents and text
embeddings. Each batch may supply deterministic noise/timesteps or leave them
to the job RNG. Supports accumulation, clipping, schedules, exact checkpoint
resume, adapter snapshots, and retention.
"""
function train!(job::WanTrainingJob; resume_from=nothing, callback=nothing)
    runtime = job.distributed
    main_process = is_main_process(runtime)
    named_parameters = wan_lora_parameters(job.model)
    isempty(named_parameters) &&
        throw(ArgumentError("Wan training model has no LoRA adapters"))
    parameters = [entry.value for entry in named_parameters]
    decay_mask = weight_decay_mask(
        [entry.name for entry in named_parameters],
        job.training.weight_decay_exclusions)
    restored_smoothed_loss = Ref{Union{Nothing,Float32}}(nothing)
    state = resume_from === nothing ?
        TrainingState(0, 0, Xoshiro(job.training.seed), AdamWState(parameters)) :
        begin
            restored = load_checkpoint(resume_from)
            _validate_resume_identity(
                restored.metadata, "wan21", job.base_model)
            _validate_resume_world_size(
                restored.metadata, distributed_world_size(runtime))
            _validate_resume_low_vram(restored.metadata, job.low_vram)
            _validate_resume_configuration(
                restored.metadata, job.training, job.flow_matching,
                wan_lora_layers(job.model))
            restored_smoothed_loss[] =
                _resume_smoothed_loss(restored.metadata)
            _restore_wan_state(restored, parameters)
        end
    state.step <= job.training.steps ||
        throw(ArgumentError("checkpoint step exceeds configured training steps"))
    optimizer = AdamW(learning_rate=job.training.learning_rate,
                      weight_decay=job.training.weight_decay)
    shadow_model = eltype(first(parameters)) === BFloat16 ?
        move_to_device(job.model, float32_values) : nothing
    dropout_enabled = any(
        entry -> entry.layer.dropout > 0f0, wan_lora_layers(job.model))
    accumulated = [_zero_state_like(parameter) for parameter in parameters]
    losses = Float32[]
    metrics = NamedTuple[]
    validations = NamedTuple[]
    metrics_path = main_process ?
        _prepare_metrics_log(job.output_dir, state.step) : nothing
    gpu_peak_memory_bytes = Int64(0)
    smoothed_loss = restored_smoothed_loss[]

    while state.step < job.training.steps
        step_started = time_ns()
        data_seconds = 0.0
        train_seconds = 0.0
        foreach(buffer -> fill!(buffer, 0f0), accumulated)
        loss_sum = 0f0
        noisy_rms_sum = 0f0
        target_rms_sum = 0f0
        active_latent_bucket = ""
        for _ in 1:job.training.gradient_accumulation
            data_started = time_ns()
            batch = job.batch(state.rng)
            batch isa WanLatentBatch ||
                throw(ArgumentError("Wan batch provider must return WanLatentBatch"))
            latents, context = batch.latents, batch.text_context
            active_latent_bucket =
                join(size(latents)[1:end - 1], 'x')
            size(latents, ndims(latents)) == size(context, 3) ||
                throw(DimensionMismatch("latent and text batch dimensions differ"))
            count = size(latents, ndims(latents))
            noise = batch.noise === nothing ?
                _device_random_normal(state.rng, latents) : batch.noise
            timesteps = batch.timesteps === nothing ?
                _constant_like(latents, sample_flow_timesteps(
                    job.flow_matching, :wan21, state.rng, count,
                    size(latents, ndims(latents) - 1))) : batch.timesteps
            flow = flow_sample(latents, noise, timesteps)
            _synchronize_training_device(flow.noisy)
            noisy_rms_sum += _tensor_rms(flow.noisy)
            target_rms_sum += _tensor_rms(flow.target)
            data_seconds += _elapsed_seconds(data_started)
            train_started = time_ns()
            dropout_seed = dropout_enabled ?
                rand(state.rng, UInt64) : UInt64(0)
            differentiated = _with_oom_diagnostics(
                    "Wan", flow.noisy, job.training, job.low_vram) do
                wan_lora_loss_and_gradients(
                    job.model, flow.noisy, timesteps, context, flow.target;
                    checkpoint_interval=
                        job.training.activation_checkpointing ?
                            job.training.checkpoint_interval : 0,
                    loss_scale=job.training.loss_scale,
                    dropout_seed=dropout_seed,
                    shadow_model=shadow_model,
                    conditioning_video=batch.conditioning_video,
                    image_features=batch.image_features)
            end
            _synchronize_training_device(first(parameters))
            train_seconds += _elapsed_seconds(train_started)
            for index in eachindex(accumulated)
                accumulated[index] .+= differentiated.gradients[index]
            end
            loss_sum += differentiated.loss
            state.micro_step += 1
        end
        differentiated = nothing
        flow = nothing
        noise = nothing
        _reclaim_training_memory(first(parameters))

        inverse_accumulation = 1f0 / job.training.gradient_accumulation
        foreach(buffer -> (buffer .*= inverse_accumulation), accumulated)
        allreduce_gradients!(runtime, accumulated)
        norm_squared = sum(sum(abs2, gradient) for gradient in accumulated)
        gradient_norm = Float32(sqrt(norm_squared))
        _assert_finite_training_step(
            "Wan", state.step + 1,
            loss_sum * inverse_accumulation, gradient_norm)
        if gradient_norm > job.training.max_gradient_norm
            scale = job.training.max_gradient_norm / gradient_norm
            foreach(buffer -> (buffer .*= scale), accumulated)
        end
        lr = learning_rate(job.training.learning_rate,
            job.training.scheduler, state.step + 1, job.training.steps)
        optimizer_started = time_ns()
        update!(optimizer, state.optimizer, parameters, accumulated;
                learning_rate=lr, weight_decay_mask=decay_mask)
        _synchronize_training_device(first(parameters))
        optimizer_seconds = _elapsed_seconds(optimizer_started)
        state.step += 1
        loss = distributed_mean_scalar(
            runtime, loss_sum * inverse_accumulation, first(parameters))
        noisy_rms = noisy_rms_sum * inverse_accumulation
        target_rms = target_rms_sum * inverse_accumulation
        error_rms = Float32(sqrt(loss))
        smoothed_loss = smoothed_loss === nothing ? loss :
            0.9f0 * smoothed_loss + 0.1f0 * loss
        push!(losses, loss)
        memory = _training_memory_metrics(first(parameters))
        gpu_peak_memory_bytes = max(
            gpu_peak_memory_bytes, memory.gpu_memory_bytes)
        current = merge((
            step=state.step, micro_step=state.micro_step, loss=loss,
            smoothed_loss=smoothed_loss,
            learning_rate=lr, gradient_norm=gradient_norm,
            adapter_parameter_norm=_adapter_parameter_norm(parameters),
            active_latent_bucket=active_latent_bucket,
            noisy_rms=noisy_rms, target_rms=target_rms,
            error_rms=error_rms,
            data_seconds=data_seconds, train_seconds=train_seconds,
            optimizer_seconds=optimizer_seconds,
            step_seconds=_elapsed_seconds(step_started),
            distributed_rank=distributed_rank(runtime),
            distributed_world_size=distributed_world_size(runtime)),
            memory, (gpu_peak_memory_bytes=gpu_peak_memory_bytes,))
        push!(metrics, current)
        main_process &&
            _append_training_metric(metrics_path, "wan21", current)
        main_process && callback !== nothing && callback(state, current)

        if main_process && job.validation.every_steps > 0 &&
           state.step % job.validation.every_steps == 0
            isempty(job.validation_batches) &&
                throw(ArgumentError(
                    "Wan validation is enabled but no validation batches were provided"))
            append!(validations, run_validation!(
                job.model, job.validation_batches, job.output_dir, state.step;
                steps=job.validation.inference_steps))
        end
        if main_process && state.step % job.checkpoint.every_steps == 0
            _save_wan_training_checkpoint(job, state, parameters, current)
        end
    end

    final_adapter = nothing
    summary = nothing
    if main_process
        mkpath(job.output_dir)
        final_adapter = joinpath(job.output_dir, "adapter-final.safetensors")
        save_wan_lora(final_adapter, job.model; base_model=job.base_model,
            metadata=Dict("training_step" => string(state.step)))
        summary = _write_run_summary(job.output_dir, "wan21", state, metrics,
                                     validations, final_adapter)
    end
    distributed_barrier!(runtime)
    (state=state, losses=losses, metrics=metrics, validations=validations,
     adapter=final_adapter, metrics_path=metrics_path, summary=summary)
end
