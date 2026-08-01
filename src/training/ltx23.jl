function ltx23_lora_loss_and_gradients(
        model::LTXVideoTransformer, latents, timesteps, context, positions,
        target; checkpoint_interval::Int=0, loss_scale::Real=1f0,
        dropout_seed::UInt64=UInt64(0),
        shadow_model=nothing)
    scale = _validated_loss_scale(loss_scale)
    named = ltx23_lora_parameters(model)
    isempty(named) &&
        throw(ArgumentError("LTX model has no trainable LoRA parameters"))
    parameters = [entry.value for entry in named]
    names = [entry.name for entry in named]

    if eltype(first(parameters)) === BFloat16
        prediction = ltx_video_transformer_forward(
            model, latents, timesteps, context, positions;
            checkpoint_interval=checkpoint_interval, training=true,
            dropout_seed=dropout_seed)
        error = float32_values(prediction) .- float32_values(target)
        loss = sum(abs2, error) / Float32(length(error))
        model32 = shadow_model === nothing ?
            move_to_device(model, float32_values) : shadow_model
        latents32 = float32_values(latents)
        timesteps32 = float32_values(timesteps)
        context32 = float32_values(context)
        target32 = float32_values(target)
        named32 = ltx23_lora_parameters(model32)
        _synchronize_lora_shadow!(named, named32)
        parameters32 = [entry.value for entry in named32]
        differentiated = Zygote.withgradient(
                Zygote.Params(parameters32)) do
            output = ltx_video_transformer_forward(
                model32, latents32, timesteps32, context32, positions;
                checkpoint_interval=checkpoint_interval, training=true,
                dropout_seed=dropout_seed)
            delta = output .- target32
            scale * sum(abs2, delta) / Float32(length(delta))
        end
        gradients = [differentiated.grad[p] for p in parameters32]
        any(isnothing, gradients) &&
            throw(ErrorException("AD returned no LTX LoRA gradient"))
        gradients = _unscale_gradients(gradients, scale)
        return (loss=Float32(loss), names=names, parameters=parameters,
                gradients=gradients)
    end

    differentiated = Zygote.withgradient(Zygote.Params(parameters)) do
        output = ltx_video_transformer_forward(
            model, latents, timesteps, context, positions;
            checkpoint_interval=checkpoint_interval, training=true,
            dropout_seed=dropout_seed)
        size(output) == size(target) ||
            throw(DimensionMismatch("LTX prediction and target shapes differ"))
        delta = float32_values(output) .- float32_values(target)
        scale * sum(abs2, delta) / Float32(length(delta))
    end
    gradients = [differentiated.grad[p] for p in parameters]
    any(isnothing, gradients) &&
        throw(ErrorException("AD returned no LTX LoRA gradient"))
    gradients = _unscale_gradients(gradients, scale)
    (loss=Float32(differentiated.val / scale), names=names,
     parameters=parameters,
     gradients=gradients)
end

function ltx23_lora_step!(
        model::LTXVideoTransformer, optimizer::AdamW, state::AdamWState,
        latents, timesteps, context, positions, target;
        learning_rate=optimizer.learning_rate,
        max_gradient_norm=Inf32, checkpoint_interval::Int=0,
        loss_scale::Real=1f0,
        dropout_seed::UInt64=UInt64(0),
        shadow_model=nothing)
    differentiated = ltx23_lora_loss_and_gradients(
        model, latents, timesteps, context, positions, target;
        checkpoint_interval=checkpoint_interval,
        loss_scale=loss_scale,
        dropout_seed=dropout_seed,
        shadow_model=shadow_model)
    norm_squared = sum(sum(abs2, gradient)
                       for gradient in differentiated.gradients)
    gradient_norm = Float32(sqrt(norm_squared))
    _assert_finite_training_step(
        "LTX-2.3", state.step + 1, differentiated.loss, gradient_norm)
    gradients = differentiated.gradients
    if isfinite(max_gradient_norm) && gradient_norm > max_gradient_norm
        scale = Float32(max_gradient_norm / gradient_norm)
        gradients = [gradient .* scale for gradient in gradients]
    end
    update!(optimizer, state, differentiated.parameters, gradients;
            learning_rate=learning_rate)
    (loss=differentiated.loss, gradient_norm=gradient_norm, state=state,
     names=differentiated.names)
end
