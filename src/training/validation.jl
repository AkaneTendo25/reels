struct WanValidationBatch{N,C,V,I}
    prompt::String
    noise::N
    text_context::C
    conditioning_video::V
    image_features::I
end
WanValidationBatch(prompt, noise, text_context;
                   conditioning_video=nothing, image_features=nothing) =
    WanValidationBatch(prompt, noise, text_context,
                       conditioning_video, image_features)

struct LTXValidationBatch{N,C,P}
    prompt::String
    noise::N
    text_context::C
    positions::P
end

"""
    flow_euler_timesteps(steps; shift=1)

Descending Euler sigma grid for rectified-flow validation, including both
endpoints. `shift > 1` applies the SD3/Wan rational time shift
`shift * sigma / (1 + (shift - 1) * sigma)`.
"""
function flow_euler_timesteps(steps::Integer; shift::Real=1)
    steps > 0 || throw(ArgumentError("validation steps must be positive"))
    isfinite(shift) && shift > 0 ||
        throw(ArgumentError("flow shift must be finite and positive"))
    sigmas = collect(range(1f0, 0f0; length=Int(steps) + 1))
    value = Float32(shift)
    value == 1f0 && return sigmas
    value .* sigmas ./ (1f0 .+ (value - 1f0) .* sigmas)
end

function _flow_euler_update(latent, velocity, delta::Real)
    size(latent) == size(velocity) ||
        throw(DimensionMismatch("validation latent and velocity shapes differ"))
    updated = float32_values(latent) .+
              Float32(delta) .* float32_values(velocity)
    cast_values(eltype(latent), updated)
end

const LTX23_DISTILLED_SIGMAS = Float32[
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
]

"""Official LTX token-shifted sigma schedule, or the exact 8-step distilled grid."""
function ltx23_sigma_schedule(steps::Integer, tokens::Integer;
                              kind::Symbol=:ltx)
    steps > 0 && tokens > 0 ||
        throw(ArgumentError("LTX schedule dimensions must be positive"))
    if kind === :distilled
        steps == 8 || throw(ArgumentError(
            "LTX-2.3 distilled schedule requires exactly 8 steps"))
        return copy(LTX23_DISTILLED_SIGMAS)
    end
    kind === :ltx ||
        throw(ArgumentError("LTX sigma schedule must be ltx or distilled"))
    sigmas = collect(range(1f0, 0f0; length=Int(steps) + 1))
    slope = (2.05f0 - 0.95f0) / (4096f0 - 1024f0)
    shift = Float32(tokens) * slope + (0.95f0 - slope * 1024f0)
    scale = exp(shift)
    for index in eachindex(sigmas)
        sigma = sigmas[index]
        sigmas[index] = iszero(sigma) ? 0f0 :
            scale / (scale + inv(sigma) - 1f0)
    end
    nonzero = findall(x -> !iszero(x), sigmas)
    stretch = (1f0 - sigmas[last(nonzero)]) / 0.9f0
    sigmas[nonzero] .= 1f0 .- (1f0 .- sigmas[nonzero]) ./ stretch
    sigmas
end

function _ltx23_tensor_std(x)
    values = float32_values(x)
    count = length(values)
    count > 1 || return 0f0
    center = sum(values) / Float32(count)
    Float32(sqrt(sum(abs2, values .- center) / Float32(count - 1)))
end

function _ltx23_x0_prediction(model, latent, sigma::Float32, context,
                              positions; negative_context=nothing,
                              guidance_scale::Real=1,
                              rescale_scale::Real=0)
    model_type = eltype(model.patchify_proj.weight)
    model_input = cast_values(model_type, latent)
    timestep = _constant_like(model_input, fill(sigma, size(model_input, 3)))
    conditional_velocity = ltx_video_transformer_forward(
        model, model_input, timestep, context, positions)
    latent32 = float32_values(latent)
    conditional_x0 = latent32 .-
        sigma .* float32_values(conditional_velocity)
    prediction = if negative_context === nothing || guidance_scale == 1
        conditional_x0
    else
        unconditional_velocity = ltx_video_transformer_forward(
            model, model_input, timestep, negative_context, positions)
        unconditional_x0 = latent32 .-
            sigma .* float32_values(unconditional_velocity)
        unconditional_x0 .+ Float32(guidance_scale) .*
            (conditional_x0 .- unconditional_x0)
    end
    if rescale_scale > 0
        predicted_std = _ltx23_tensor_std(prediction)
        if predicted_std > 1f-6
            factor = _ltx23_tensor_std(conditional_x0) / predicted_std
            factor = Float32(rescale_scale) * factor +
                     (1f0 - Float32(rescale_scale))
            prediction = prediction .* factor
        end
    end
    prediction
end

function _ltx23_res2s_midpoint(sample, denoised, sigma::Float32,
                               next_sigma::Float32)
    next_sigma <= 0f0 && return nothing
    h = -log(next_sigma / sigma)
    z = -0.5f0 * h
    phi1 = abs(z) < 1f-5 ? 1f0 + z / 2f0 + z^2 / 6f0 : expm1(z) / z
    midpoint = sample .+ (0.5f0 * h * phi1) .* (denoised .- sample)
    midpoint, Float32(exp(log(sigma) - 0.5f0 * h))
end

function _ltx23_res2s_step(sample, first_x0, second_x0,
                           sigma::Float32, next_sigma::Float32)
    next_sigma <= 0f0 && return first_x0
    h = -log(next_sigma / sigma)
    z = -h
    if abs(z) < 1f-5
        phi1 = 1f0 + z / 2f0 + z^2 / 6f0
        phi2 = 0.5f0 + z / 6f0 + z^2 / 24f0
    else
        phi1 = expm1(z) / z
        phi2 = (expm1(z) - z) / z^2
    end
    second_weight = phi2 / 0.5f0
    first_weight = phi1 - second_weight
    sample .+ h .* (first_weight .* (first_x0 .- sample) .+
                    second_weight .* (second_x0 .- sample))
end

function ltx23_validation_sample(model::LTXVideoTransformer, noise,
                                 context, positions; steps::Integer=15,
                                 sigma_schedule::Symbol=:ltx,
                                 sampler::Symbol=:res2s,
                                 negative_context=nothing,
                                 guidance_scale::Real=1,
                                 rescale_scale::Real=0,
                                 callback=nothing)
    size(noise, 3) == size(context, 3) == size(positions, 3) ||
        throw(DimensionMismatch("LTX validation batch dimensions differ"))
    negative_context === nothing ||
        size(negative_context, 3) == size(noise, 3) ||
        throw(DimensionMismatch("LTX negative-context batch dimension differs"))
    negative_context === nothing && guidance_scale != 1 &&
        throw(ArgumentError("LTX guidance requires a negative context"))
    sampler in (:euler, :res2s) ||
        throw(ArgumentError("LTX sampler must be euler or res2s"))
    0 <= rescale_scale <= 1 ||
        throw(ArgumentError("LTX rescale scale must be in [0, 1]"))
    latent = float32_values(noise)
    schedule = ltx23_sigma_schedule(
        steps, size(noise, 2); kind=sigma_schedule)
    for index in 1:length(schedule)-1
        sigma, next_sigma = schedule[index], schedule[index + 1]
        x0 = _ltx23_x0_prediction(
            model, latent, sigma, context, positions;
            negative_context=negative_context,
            guidance_scale=guidance_scale, rescale_scale=rescale_scale)
        if sampler === :res2s
            midpoint = _ltx23_res2s_midpoint(latent, x0, sigma, next_sigma)
            if midpoint === nothing
                latent = x0
            else
                midpoint_latent, midpoint_sigma = midpoint
                midpoint_x0 = _ltx23_x0_prediction(
                    model, midpoint_latent, midpoint_sigma, context, positions;
                    negative_context=negative_context,
                    guidance_scale=guidance_scale,
                    rescale_scale=rescale_scale)
                latent = _ltx23_res2s_step(
                    latent, x0, midpoint_x0, sigma, next_sigma)
            end
        else
            direction = (latent .- x0) ./ sigma
            latent = latent .+ (next_sigma - sigma) .* direction
        end
        callback === nothing || callback(index, next_sigma, latent)
    end
    latent
end

function wan_validation_sample(model::WanTransformer, noise,
                               context; steps::Integer=20,
                               flow_shift::Real=5,
                               negative_context=nothing,
                               guidance_scale::Real=1,
                               conditioning_video=nothing,
                               image_features=nothing,
                               callback=nothing)
    size(noise, 5) == size(context, 3) ||
        throw(DimensionMismatch("Wan validation batch dimensions differ"))
    isfinite(guidance_scale) && guidance_scale >= 0 ||
        throw(ArgumentError(
            "Wan guidance scale must be finite and nonnegative"))
    negative_context === nothing || size(noise, 5) ==
        size(negative_context, 3) ||
        throw(DimensionMismatch(
            "Wan negative-context batch dimension differs"))
    negative_context === nothing && guidance_scale != 1 &&
        throw(ArgumentError(
            "Wan guidance scale requires a negative prompt context"))
    # Upstream Wan keeps the scheduler sample in FP32 even when DiT weights and
    # activations use BF16. Re-quantizing the accumulated sample after every
    # Euler step causes substantial trajectory drift over production schedules.
    latent = float32_values(noise)
    schedule = flow_euler_timesteps(steps; shift=flow_shift)
    for index in 1:length(schedule)-1
        sigma = _constant_like(latent,
            fill(schedule[index], size(latent, 5)))
        model_input = cast_values(eltype(model.patch_weight), latent)
        velocity = wan_transformer_forward(model, model_input, sigma, context;
            conditioning_video=conditioning_video,
            image_features=image_features)
        if negative_context !== nothing && guidance_scale != 1
            unconditional = wan_transformer_forward(
                model, model_input, sigma, negative_context;
                conditioning_video=conditioning_video,
                image_features=image_features)
            guided = float32_values(unconditional) .+
                Float32(guidance_scale) .* (
                    float32_values(velocity) .-
                    float32_values(unconditional))
            velocity = cast_values(eltype(velocity), guided)
        end
        latent = _flow_euler_update(
            latent, velocity, schedule[index + 1] - schedule[index])
        callback === nothing ||
            callback(index, schedule[index + 1], latent)
    end
    latent
end

function _with_adapters_disabled(operation, model, layers)
    enabled = [entry.layer.enabled for entry in layers]
    try
        foreach(entry -> set_adapter_enabled!(entry.layer, false), layers)
        operation()
    finally
        for (entry, state) in zip(layers, enabled)
            set_adapter_enabled!(entry.layer, state)
        end
    end
end

function ltx23_validation_comparison(model::LTXVideoTransformer, noise,
                                     context, positions; steps::Integer=20)
    layers = ltx23_lora_layers(model)
    isempty(layers) &&
        throw(ArgumentError("LTX validation comparison requires adapters"))
    enabled = ltx23_validation_sample(
        model, noise, context, positions; steps=steps)
    disabled = _with_adapters_disabled(model, layers) do
        ltx23_validation_sample(
            model, noise, context, positions; steps=steps)
    end
    delta = sum(abs, float32_values(enabled) .-
                     float32_values(disabled)) / length(enabled)
    (enabled=enabled, disabled=disabled, mean_absolute_delta=Float32(delta))
end

function wan_validation_comparison(model::WanTransformer, noise,
                                   context; steps::Integer=20,
                                   flow_shift::Real=5,
                                   negative_context=nothing,
                                   guidance_scale::Real=1,
                                   conditioning_video=nothing,
                                   image_features=nothing)
    layers = wan_lora_layers(model)
    isempty(layers) &&
        throw(ArgumentError("Wan validation comparison requires adapters"))
    enabled = wan_validation_sample(model, noise, context; steps=steps,
        flow_shift=flow_shift, negative_context=negative_context,
        guidance_scale=guidance_scale,
        conditioning_video=conditioning_video,
        image_features=image_features)
    disabled = _with_adapters_disabled(model, layers) do
        wan_validation_sample(model, noise, context; steps=steps,
            flow_shift=flow_shift, negative_context=negative_context,
            guidance_scale=guidance_scale,
            conditioning_video=conditioning_video,
            image_features=image_features)
    end
    delta = sum(abs, float32_values(enabled) .-
                     float32_values(disabled)) / length(enabled)
    (enabled=enabled, disabled=disabled, mean_absolute_delta=Float32(delta))
end

function _validation_stem(prompt::AbstractString)
    bytes2hex(sha256(String(prompt)))[1:16]
end

function _write_validation_artifact(output_dir::AbstractString,
                                    family::AbstractString,
                                    step::Integer,
                                    prompt::AbstractString,
                                    comparison)
    directory = joinpath(output_dir, "validation", "step-$(Int(step))")
    path = joinpath(directory, "$family-$(_validation_stem(prompt)).safetensors")
    metadata = Dict{String,String}(
        "format" => "reels-validation-latents-v1",
        "model_family" => String(family),
        "step" => string(step),
        "prompt" => String(prompt),
        "mean_absolute_delta" => string(comparison.mean_absolute_delta),
    )
    write_safetensors(path, Dict(
        "adapter_enabled" => Array(comparison.enabled),
        "adapter_disabled" => Array(comparison.disabled),
    ); metadata=metadata)
    path
end

function run_validation!(model::WanTransformer,
                         batches::AbstractVector{<:WanValidationBatch},
                         output_dir::AbstractString, step::Integer;
                         steps::Integer=20)
    map(batches) do batch
        comparison = wan_validation_comparison(
            model, batch.noise, batch.text_context; steps=steps,
            conditioning_video=batch.conditioning_video,
            image_features=batch.image_features)
        path = _write_validation_artifact(
            output_dir, "wan21", step, batch.prompt, comparison)
        (step=Int(step), prompt=batch.prompt, path=path,
         mean_absolute_delta=comparison.mean_absolute_delta)
    end
end

function run_validation!(model::LTXVideoTransformer,
                         batches::AbstractVector{<:LTXValidationBatch},
                         output_dir::AbstractString, step::Integer;
                         steps::Integer=20)
    map(batches) do batch
        comparison = ltx23_validation_comparison(
            model, batch.noise, batch.text_context, batch.positions;
            steps=steps)
        path = _write_validation_artifact(
            output_dir, "ltx23", step, batch.prompt, comparison)
        (step=Int(step), prompt=batch.prompt, path=path,
         mean_absolute_delta=comparison.mean_absolute_delta)
    end
end
