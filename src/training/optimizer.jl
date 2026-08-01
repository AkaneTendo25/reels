Base.@kwdef struct AdamW
    learning_rate::Float32 = 1f-4
    beta1::Float32 = 0.9f0
    beta2::Float32 = 0.999f0
    epsilon::Float32 = 1f-8
    weight_decay::Float32 = 0.01f0
end

mutable struct AdamWState{M<:AbstractVector,V<:AbstractVector,P<:AbstractVector}
    step::Int
    m::M
    v::V
    master::P
end

_zero_state_like(p) = fill!(similar(p, Float32), 0f0)
function _master_state_like(p)
    master = similar(p, Float32)
    copyto!(master, float32_values(p))
    master
end
AdamWState(params) = AdamWState(0, [_zero_state_like(p) for p in params],
    [_zero_state_like(p) for p in params],
    [_master_state_like(p) for p in params])

"""
    weight_decay_mask(names, exclusions)

Return one Boolean per stable parameter name. `false` marks a name matched by
any Julia regular expression in `exclusions`, so AdamW leaves that parameter
out of decoupled weight decay while still applying its gradient update.
"""
function weight_decay_mask(names, exclusions)
    patterns = Regex.(exclusions)
    [!any(pattern -> occursin(pattern, name), patterns) for name in names]
end

function learning_rate(base::Real, scheduler::SchedulerConfig, step::Int, total::Int)
    warm = scheduler.warmup_steps
    step <= warm && warm > 0 && return Float32(base * step / warm)
    progress = Float32(clamp((step - warm) / max(total - warm, 1), 0, 1))
    scheduler.kind === :constant && return Float32(base)
    scheduler.kind === :linear && return Float32(base) * (1f0 - progress)
    scheduler.kind === :cosine && return Float32(base) * (1f0 + cospi(progress)) / 2f0
    throw(ArgumentError("unknown scheduler $(scheduler.kind)"))
end

function update!(opt::AdamW, state::AdamWState, params, grads;
                 learning_rate=opt.learning_rate,
                 weight_decay_mask=nothing)
    length(params) == length(grads) == length(state.m) ==
        length(state.v) == length(state.master) ||
        throw(DimensionMismatch("parameter, gradient, and optimizer state lengths differ"))
    weight_decay_mask === nothing ||
        length(weight_decay_mask) == length(params) ||
        throw(DimensionMismatch(
            "weight-decay mask and parameter lengths differ"))
    state.step += 1
    correction1 = 1f0 - opt.beta1^state.step
    correction2 = 1f0 - opt.beta2^state.step
    for i in eachindex(params)
        p, g, master = params[i], float32_values(grads[i]), state.master[i]
        state.m[i] .= opt.beta1 .* state.m[i] .+ (1f0 - opt.beta1) .* g
        state.v[i] .= opt.beta2 .* state.v[i] .+ (1f0 - opt.beta2) .* abs2.(g)
        decay = weight_decay_mask === nothing || weight_decay_mask[i] ?
            opt.weight_decay : 0f0
        master .-= learning_rate .* ((state.m[i] ./ correction1) ./
            (sqrt.(state.v[i] ./ correction2) .+ opt.epsilon) .+
            decay .* master)
        copyto!(p, cast_values(eltype(p), master))
    end
    state
end
