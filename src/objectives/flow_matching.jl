struct FlowSample{A,T}
    noisy::A
    target::A
    timestep::T
end

uniform_timestep(rng::AbstractRNG, n::Integer=1) = rand(rng, Float32, n)

ltx_timestep_shift(sequence_length::Integer;
                   min_tokens=1024, max_tokens=4096,
                   min_shift=0.95f0, max_shift=2.05f0) =
    Float32(min_shift + (max_shift - min_shift) *
            (sequence_length - min_tokens) / (max_tokens - min_tokens))

"""
Official LTX trainer shifted logit-normal timestep sampler.
"""
function shifted_logit_normal_timestep(
        rng::AbstractRNG, count::Integer, sequence_length::Integer;
        standard_deviation=1f0, epsilon=1f-3,
        uniform_probability=0.1f0)
    count > 0 && sequence_length > 0 ||
        throw(ArgumentError("timestep sample dimensions must be positive"))
    standard_deviation > 0 ||
        throw(ArgumentError("timestep standard deviation must be positive"))
    0 < epsilon < 0.5 ||
        throw(ArgumentError("timestep epsilon must be in (0, 0.5)"))
    0 <= uniform_probability <= 1 ||
        throw(ArgumentError("uniform timestep probability must be in [0, 1]"))
    logistic(x) = 1f0 / (1f0 + exp(-Float32(x)))
    std = Float32(standard_deviation)
    eps = Float32(epsilon)
    mu = ltx_timestep_shift(sequence_length)
    samples = logistic.(randn(rng, Float32, Int(count)) .* std .+ mu)
    upper = logistic(mu + 3.0902f0 * std)
    lower = logistic(mu - 2.5758f0 * std)
    stretched = (samples .- lower) ./ (upper - lower)
    stretched = ifelse.(stretched .>= eps, stretched,
                        2f0 * eps .- stretched)
    stretched = clamp.(stretched, 0f0, 1f0)
    uniform = (1f0 - eps) .* rand(rng, Float32, Int(count)) .+ eps
    selector = rand(rng, Float32, Int(count))
    ifelse.(selector .> Float32(uniform_probability),
            stretched, uniform)
end

function sample_flow_timesteps(config::FlowMatchingConfig, family::Symbol,
                               rng::AbstractRNG, count::Integer,
                               sequence_length::Integer)
    mode = config.timestep_sampling === :auto ?
        (family === :ltx23 ? :shifted_logit_normal : :uniform) :
        config.timestep_sampling
    mode === :uniform && return uniform_timestep(rng, count)
    mode === :shifted_logit_normal && return
        shifted_logit_normal_timestep(rng, count, sequence_length;
            standard_deviation=config.standard_deviation,
            epsilon=config.epsilon,
            uniform_probability=config.uniform_probability)
    throw(ArgumentError("unsupported timestep sampling mode $mode"))
end

function flow_sample(latents::AbstractArray, noise::AbstractArray, timestep)
    size(latents) == size(noise) || throw(DimensionMismatch("latent and noise shapes differ"))
    if eltype(latents) === BFloat16
        latent32 = float32_values(latents)
        noise32 = float32_values(noise)
        if timestep isa AbstractVector
            length(timestep) == size(latents, ndims(latents)) ||
                throw(DimensionMismatch("timestep count and latent batch differ"))
            shape = (ntuple(_ -> 1, ndims(latents) - 1)..., length(timestep))
            timestep32 = eltype(timestep) === BFloat16 ?
                float32_values(timestep) : Float32.(timestep)
            t = reshape(timestep32, shape)
            noisy = bfloat16_values(
                (1f0 .- t) .* latent32 .+ t .* noise32)
            target = bfloat16_values(noise32 .- latent32)
            return FlowSample(noisy, target, timestep)
        end
        t = Float32(timestep)
        return FlowSample(
            bfloat16_values((1f0 - t) .* latent32 .+ t .* noise32),
            bfloat16_values(noise32 .- latent32), timestep)
    end
    if timestep isa AbstractVector
        length(timestep) == size(latents, ndims(latents)) ||
            throw(DimensionMismatch("timestep count and latent batch differ"))
        shape = (ntuple(_ -> 1, ndims(latents) - 1)..., length(timestep))
        t = reshape(eltype(latents).(timestep), shape)
        one_t = one(eltype(latents))
        noisy = (one_t .- t) .* latents .+ t .* noise
        return FlowSample(noisy, noise .- latents, timestep)
    end
    t = eltype(latents)(timestep)
    FlowSample((one(eltype(latents)) - t) .* latents .+ t .* noise,
               noise .- latents, t)
end
