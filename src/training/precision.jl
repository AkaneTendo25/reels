function _validated_loss_scale(value::Real)
    scale = Float32(value)
    isfinite(scale) && scale > 0f0 ||
        throw(ArgumentError("loss_scale must be finite and positive"))
    scale
end

function _unscale_gradients(gradients, loss_scale::Float32)
    loss_scale == 1f0 && return gradients
    inverse_scale = 1f0 / loss_scale
    [float32_values(gradient) .* inverse_scale for gradient in gradients]
end
