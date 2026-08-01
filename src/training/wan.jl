function _synchronize_lora_shadow!(source, shadow)
    length(source) == length(shadow) ||
        throw(DimensionMismatch("LoRA shadow parameter count differs"))
    for (source_entry, shadow_entry) in zip(source, shadow)
        source_entry.name == shadow_entry.name ||
            throw(ArgumentError("LoRA shadow parameter names differ"))
        size(source_entry.value) == size(shadow_entry.value) ||
            throw(DimensionMismatch("LoRA shadow parameter shape differs"))
        copyto!(shadow_entry.value, float32_values(source_entry.value))
    end
    shadow
end

"""
    wan_lora_loss_and_gradients(model, video, timesteps, text_context, target)

Compute mean-squared flow loss and reverse-mode gradients for every injected
Wan LoRA tensor. Frozen base weights participate in forward/backward
propagation but are not included in the differentiation parameter set.
"""
function wan_lora_loss_and_gradients(model::WanTransformer, video, timesteps,
                                     text_context, target;
                                     checkpoint_interval::Int=0,
                                     loss_scale::Real=1f0,
                                     dropout_seed::UInt64=UInt64(0),
                                     shadow_model=nothing,
                                     conditioning_video=nothing,
                                     image_features=nothing)
    scale = _validated_loss_scale(loss_scale)
    named_parameters = wan_lora_parameters(model)
    isempty(named_parameters) &&
        throw(ArgumentError("Wan model has no trainable LoRA parameters"))
    parameters = [entry.value for entry in named_parameters]
    names = [entry.name for entry in named_parameters]

    # PTX does not implement every scalar BFloat16 operation needed by
    # Zygote's generated pullbacks. Keep the stored model and real forward in
    # BF16, then recompute the backward pass from an FP32 view of those
    # quantized values. Gradients and AdamW masters consequently remain FP32.
    if eltype(first(parameters)) === BFloat16
        prediction = wan_transformer_forward(
            model, video, timesteps, text_context;
            checkpoint_interval=checkpoint_interval,
            conditioning_video=conditioning_video,
            image_features=image_features, training=true,
            dropout_seed=dropout_seed)
        size(prediction) == size(target) ||
            throw(DimensionMismatch("Wan prediction and target shapes differ"))
        error = float32_values(prediction) .- float32_values(target)
        loss = sum(abs2, error) / Float32(length(error))

        model32 = shadow_model === nothing ?
            move_to_device(model, float32_values) : shadow_model
        video32 = float32_values(video)
        timesteps32 = float32_values(timesteps)
        text_context32 = float32_values(text_context)
        target32 = float32_values(target)
        conditioning32 = conditioning_video === nothing ? nothing :
            float32_values(conditioning_video)
        image32 = image_features === nothing ? nothing :
            float32_values(image_features)
        named_parameters32 = wan_lora_parameters(model32)
        _synchronize_lora_shadow!(named_parameters, named_parameters32)
        parameters32 = [entry.value for entry in named_parameters32]
        parameter_set32 = Zygote.Params(parameters32)
        result32 = Zygote.withgradient(parameter_set32) do
            prediction32 = wan_transformer_forward(
                model32, video32, timesteps32, text_context32;
                checkpoint_interval=checkpoint_interval,
                conditioning_video=conditioning32,
                image_features=image32, training=true,
                dropout_seed=dropout_seed)
            error32 = prediction32 .- target32
            scale * sum(abs2, error32) / Float32(length(error32))
        end
        gradients = [result32.grad[parameter] for parameter in parameters32]
        any(isnothing, gradients) &&
            throw(ErrorException("AD returned no gradient for a Wan LoRA parameter"))
        gradients = _unscale_gradients(gradients, scale)
        return (loss=Float32(loss), names=names, parameters=parameters,
                gradients=gradients)
    end

    parameter_set = Zygote.Params(parameters)
    result = Zygote.withgradient(parameter_set) do
        prediction = wan_transformer_forward(model, video, timesteps,
                                             text_context;
            checkpoint_interval=checkpoint_interval,
            conditioning_video=conditioning_video,
            image_features=image_features, training=true,
            dropout_seed=dropout_seed)
        size(prediction) == size(target) ||
            throw(DimensionMismatch("Wan prediction and target shapes differ"))
        error = float32_values(prediction) .- float32_values(target)
        scale * sum(abs2, error) / Float32(length(error))
    end
    gradients = [result.grad[parameter] for parameter in parameters]
    any(isnothing, gradients) &&
        throw(ErrorException("AD returned no gradient for a Wan LoRA parameter"))
    gradients = _unscale_gradients(gradients, scale)
    (loss=Float32(result.val / scale), names=names, parameters=parameters,
     gradients=gradients)
end

"""
    wan_lora_step!(model, optimizer, state, video, timesteps, text, target)

Run one complete Wan LoRA forward/backward/AdamW update. Optimizer moments stay
on the parameter device. Returns the scalar loss, gradient norm before clipping,
and updated state.
"""
function wan_lora_step!(model::WanTransformer, optimizer::AdamW,
                        state::AdamWState, video, timesteps, text_context,
                        target; learning_rate=optimizer.learning_rate,
                        max_gradient_norm=Inf32,
                        checkpoint_interval::Int=0,
                        loss_scale::Real=1f0,
                        dropout_seed::UInt64=UInt64(0),
                        shadow_model=nothing,
                        conditioning_video=nothing,
                        image_features=nothing)
    differentiated = wan_lora_loss_and_gradients(
        model, video, timesteps, text_context, target;
        checkpoint_interval=checkpoint_interval,
        loss_scale=loss_scale,
        dropout_seed=dropout_seed,
        shadow_model=shadow_model,
        conditioning_video=conditioning_video,
        image_features=image_features)
    length(state.m) == length(differentiated.parameters) ||
        throw(DimensionMismatch("optimizer state does not match Wan LoRA parameters"))
    norm_squared = sum(sum(abs2, gradient)
                       for gradient in differentiated.gradients)
    gradient_norm = Float32(sqrt(norm_squared))
    _assert_finite_training_step(
        "Wan", state.step + 1, differentiated.loss, gradient_norm)
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
