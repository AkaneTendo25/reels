struct LTXLatentBatch{L,C,P,N,T}
    latents::L
    text_context::C
    positions::P
    noise::N
    timesteps::T
end

LTXLatentBatch(latents, text_context, positions;
               noise=nothing, timesteps=nothing) =
    LTXLatentBatch(latents, text_context, positions, noise, timesteps)

Base.@kwdef struct LTXTrainingJob{M,F,V,D}
    model::M
    batch::F
    training::TrainingConfig
    output_dir::String
    checkpoint::CheckpointConfig = CheckpointConfig()
    base_model::String = ""
    flow_matching::FlowMatchingConfig = FlowMatchingConfig()
    validation::ValidationConfig = ValidationConfig()
    validation_batches::V = LTXValidationBatch[]
    distributed::D = SingleProcessRuntime()
    low_vram::LowVRAMConfig = LowVRAMConfig()
end

function _prune_ltx23_checkpoints!(job::LTXTrainingJob)
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

function _save_ltx23_training_checkpoint(job, state, parameters, metric)
    mkpath(job.output_dir)
    step = state.step
    checkpoint = joinpath(job.output_dir, "checkpoint-$step.reels")
    adapter = joinpath(job.output_dir, "adapter-$step.safetensors")
    metadata = _training_checkpoint_metadata(
        "ltx23", job.base_model, job.training, job.checkpoint,
        job.validation, job.flow_matching, job.low_vram,
        ltx23_lora_layers(job.model), metric)
    save_checkpoint(checkpoint, state, parameters; metadata=metadata)
    save_ltx23_lora(adapter, job.model; base_model=job.base_model,
        metadata=Dict("training_step" => string(step)))
    _prune_ltx23_checkpoints!(job)
    (checkpoint=checkpoint, adapter=adapter)
end

function train!(job::LTXTrainingJob; resume_from=nothing, callback=nothing)
    runtime = job.distributed
    main_process = is_main_process(runtime)
    named = ltx23_lora_parameters(job.model)
    isempty(named) &&
        throw(ArgumentError("LTX training model has no LoRA adapters"))
    parameters = [entry.value for entry in named]
    decay_mask = weight_decay_mask(
        [entry.name for entry in named],
        job.training.weight_decay_exclusions)
    restored_smoothed_loss = Ref{Union{Nothing,Float32}}(nothing)
    state = resume_from === nothing ?
        TrainingState(0, 0, Xoshiro(job.training.seed),
                      AdamWState(parameters)) :
        begin
            restored = load_checkpoint(resume_from)
            _validate_resume_identity(
                restored.metadata, "ltx23", job.base_model)
            _validate_resume_world_size(
                restored.metadata, distributed_world_size(runtime))
            _validate_resume_low_vram(restored.metadata, job.low_vram)
            _validate_resume_configuration(
                restored.metadata, job.training, job.flow_matching,
                ltx23_lora_layers(job.model))
            restored_smoothed_loss[] =
                _resume_smoothed_loss(restored.metadata)
            _restore_wan_state(restored, parameters)
        end
    state.step <= job.training.steps ||
        throw(ArgumentError("checkpoint step exceeds configured LTX steps"))
    optimizer = AdamW(learning_rate=job.training.learning_rate,
                      weight_decay=job.training.weight_decay)
    shadow_model = eltype(first(parameters)) === BFloat16 ?
        move_to_device(job.model, float32_values) : nothing
    dropout_enabled = any(
        entry -> entry.layer.dropout > 0f0, ltx23_lora_layers(job.model))
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
            batch isa LTXLatentBatch ||
                throw(ArgumentError("LTX batch provider must return LTXLatentBatch"))
            active_latent_bucket =
                join(size(batch.latents)[1:end - 1], 'x')
            count = size(batch.latents, 3)
            size(batch.text_context, 3) == size(batch.positions, 3) == count ||
                throw(DimensionMismatch("LTX batch dimensions differ"))
            noise = batch.noise === nothing ?
                _device_random_normal(state.rng, batch.latents) : batch.noise
            timesteps = batch.timesteps === nothing ?
                _constant_like(batch.latents, sample_flow_timesteps(
                    job.flow_matching, :ltx23, state.rng, count,
                    size(batch.latents, 2))) :
                batch.timesteps
            flow = flow_sample(batch.latents, noise, timesteps)
            _synchronize_training_device(flow.noisy)
            noisy_rms_sum += _tensor_rms(flow.noisy)
            target_rms_sum += _tensor_rms(flow.target)
            data_seconds += _elapsed_seconds(data_started)
            train_started = time_ns()
            dropout_seed = dropout_enabled ?
                rand(state.rng, UInt64) : UInt64(0)
            differentiated = _with_oom_diagnostics(
                    "LTX-2.3", flow.noisy, job.training, job.low_vram) do
                ltx23_lora_loss_and_gradients(
                    job.model, flow.noisy, timesteps, batch.text_context,
                    batch.positions, flow.target;
                    checkpoint_interval=
                        job.training.activation_checkpointing ?
                            job.training.checkpoint_interval : 0,
                    loss_scale=job.training.loss_scale,
                    dropout_seed=dropout_seed,
                    shadow_model=shadow_model)
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
        inverse = 1f0 / job.training.gradient_accumulation
        foreach(buffer -> (buffer .*= inverse), accumulated)
        allreduce_gradients!(runtime, accumulated)
        norm_squared = sum(sum(abs2, gradient) for gradient in accumulated)
        gradient_norm = Float32(sqrt(norm_squared))
        _assert_finite_training_step(
            "LTX-2.3", state.step + 1, loss_sum * inverse, gradient_norm)
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
            runtime, loss_sum * inverse, first(parameters))
        noisy_rms = noisy_rms_sum * inverse
        target_rms = target_rms_sum * inverse
        error_rms = Float32(sqrt(loss))
        smoothed_loss = smoothed_loss === nothing ? loss :
            0.9f0 * smoothed_loss + 0.1f0 * loss
        push!(losses, loss)
        memory = _training_memory_metrics(first(parameters))
        gpu_peak_memory_bytes = max(
            gpu_peak_memory_bytes, memory.gpu_memory_bytes)
        metric = merge((
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
        push!(metrics, metric)
        main_process &&
            _append_training_metric(metrics_path, "ltx23", metric)
        main_process && callback !== nothing && callback(state, metric)
        if main_process && job.validation.every_steps > 0 &&
           state.step % job.validation.every_steps == 0
            isempty(job.validation_batches) &&
                throw(ArgumentError(
                    "LTX validation is enabled but no validation batches were provided"))
            append!(validations, run_validation!(
                job.model, job.validation_batches, job.output_dir, state.step;
                steps=job.validation.inference_steps))
        end
        main_process && state.step % job.checkpoint.every_steps == 0 &&
            _save_ltx23_training_checkpoint(job, state, parameters, metric)
    end
    adapter = nothing
    summary = nothing
    if main_process
        mkpath(job.output_dir)
        adapter = joinpath(job.output_dir, "adapter-final.safetensors")
        save_ltx23_lora(adapter, job.model; base_model=job.base_model,
            metadata=Dict("training_step" => string(state.step)))
        summary = _write_run_summary(job.output_dir, "ltx23", state, metrics,
                                     validations, adapter)
    end
    distributed_barrier!(runtime)
    (state=state, losses=losses, metrics=metrics, validations=validations,
     adapter=adapter, metrics_path=metrics_path, summary=summary)
end
