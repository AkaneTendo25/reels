struct VAEConv3D{W<:AbstractArray,B<:AbstractVector}
    weight::W
    bias::B
    stride::NTuple{3,Int}
    padding::NTuple{3,Int}
end

function VAEConv3D(weight::AbstractArray, bias::AbstractVector;
                   stride=(1, 1, 1), padding=(0, 0, 0))
    ndims(weight) == 5 ||
        throw(DimensionMismatch("VAE convolution weight must be 5-dimensional"))
    size(weight, 1) == length(bias) ||
        throw(DimensionMismatch("VAE convolution bias size differs"))
    VAEConv3D(weight, bias, Tuple(Int.(stride)), Tuple(Int.(padding)))
end

"""PyTorch-compatible 3D cross-correlation on `(C,T,H,W,B)` tensors."""
function vae_conv3d(layer::VAEConv3D, x::AbstractArray;
                    temporal_left::Integer=2 * layer.padding[1],
                    temporal_right::Integer=0,
                    spatial_padding=nothing)
    ndims(x) == 5 ||
        throw(DimensionMismatch("VAE convolution input must be 5-dimensional"))
    size(x, 1) == size(layer.weight, 2) ||
        throw(DimensionMismatch("VAE convolution input channels differ"))
    _, ph, pw = layer.padding
    spatial = spatial_padding === nothing ? (ph, ph, pw, pw) :
        Tuple(Int.(spatial_padding))
    pad = (Int(temporal_left), Int(temporal_right), spatial...)
    input = permutedims(x, (2, 3, 4, 1, 5))
    weight = permutedims(layer.weight, (3, 4, 5, 2, 1))
    # SafeTensors restoration exposes source-framework kernel coordinates.
    # Reversing explicitly preserves PyTorch parity while keeping NNlib's
    # cross-correlation path eligible for cuDNN (its `flipped=true` fallback
    # uses host-style im2col for CuArrays).
    kernel = reverse(weight; dims=(1, 2, 3))
    output = NNlib.conv(input, kernel; pad=pad, stride=layer.stride)
    output .+= reshape(layer.bias, 1, 1, 1, :, 1)
    permutedims(output, (4, 1, 2, 3, 5))
end

(layer::VAEConv3D)(x::AbstractArray; kwargs...) =
    vae_conv3d(layer, x; kwargs...)

struct VAERMSNorm{G<:AbstractVector}
    gamma::G
end

function vae_rmsnorm(norm::VAERMSNorm, x::AbstractArray)
    size(x, 1) == length(norm.gamma) ||
        throw(DimensionMismatch("VAE RMSNorm feature dimension differs"))
    x32 = float32_values(x)
    magnitude = sqrt.(sum(abs2, x32; dims=1))
    denominator = max.(magnitude, 1f-12)
    normalized = eltype(x).(
        x32 ./ denominator .* sqrt(Float32(size(x, 1))))
    normalized .* reshape(norm.gamma, :, ntuple(_ -> 1, ndims(x) - 1)...)
end

vae_silu(x) = x ./ (one(eltype(x)) .+ exp.(-x))

struct VAEResidualBlock
    norm1::VAERMSNorm
    conv1::VAEConv3D
    norm2::VAERMSNorm
    conv2::VAEConv3D
    shortcut::Union{Nothing,VAEConv3D}
end

function vae_residual_forward(block::VAEResidualBlock, x::AbstractArray)
    residual = block.shortcut === nothing ? x : block.shortcut(x)
    hidden = block.conv1(vae_silu(vae_rmsnorm(block.norm1, x)))
    hidden = block.conv2(vae_silu(vae_rmsnorm(block.norm2, hidden)))
    hidden .+ residual
end

struct VAEAttentionBlock
    norm::VAERMSNorm
    qkv::VAEConv3D
    projection::VAEConv3D
end

function vae_attention_forward(block::VAEAttentionBlock, x::AbstractArray)
    channels, frames, height, width, batch = size(x)
    qkv = block.qkv(vae_rmsnorm(block.norm, x))
    q = view(qkv, 1:channels, :, :, :, :)
    k = view(qkv, channels + 1:2channels, :, :, :, :)
    v = view(qkv, 2channels + 1:3channels, :, :, :, :)
    tokens = height * width
    combined_batch = frames * batch
    canonical(array) = reshape(permutedims(array, (1, 3, 4, 2, 5)),
        channels, tokens, 1, combined_batch)
    attended =
        memory_efficient_attention(canonical(q), canonical(k), canonical(v)).output
    attended = permutedims(
        reshape(attended, channels, height, width, frames, batch),
        (1, 4, 2, 3, 5))
    x .+ block.projection(attended)
end

struct VAEDownsample
    spatial::VAEConv3D
    temporal::Union{Nothing,VAEConv3D}
end

struct VAEUpsample
    spatial::VAEConv3D
    temporal::Union{Nothing,VAEConv3D}
end

mutable struct VAEDecodeFeatureCache
    values::Vector{Any}
    index::Int
end

VAEDecodeFeatureCache() = VAEDecodeFeatureCache(Any[], 0)

function _next_vae_cache!(cache::VAEDecodeFeatureCache)
    cache.index += 1
    while length(cache.values) < cache.index
        push!(cache.values, nothing)
    end
    cache.index, cache.values[cache.index]
end

function _vae_cache_tail(x::AbstractArray, previous)
    frames = size(x, 2)
    frames > 0 || throw(DimensionMismatch("VAE cache input has no frames"))
    tail = copy(view(x, :, max(1, frames - 1):frames, :, :, :))
    size(tail, 2) == 2 && return tail
    if previous isa AbstractArray
        return cat(view(previous, :, size(previous, 2):size(previous, 2),
                    :, :, :), tail; dims=2)
    elseif previous === :rep
        padding = similar(tail)
        fill!(padding, zero(eltype(padding)))
        return cat(padding, tail; dims=2)
    end
    tail
end

function _vae_cached_conv3d(layer::VAEConv3D, x::AbstractArray,
                            cache::VAEDecodeFeatureCache)
    index, previous = _next_vae_cache!(cache)
    cache.values[index] = _vae_cache_tail(x, previous)
    temporal_left = 2 * layer.padding[1]
    if previous isa AbstractArray && temporal_left > 0
        input = cat(previous, x; dims=2)
        temporal_left = max(0, temporal_left - size(previous, 2))
        return vae_conv3d(layer, input; temporal_left=temporal_left)
    end
    vae_conv3d(layer, x)
end

function vae_residual_forward(block::VAEResidualBlock, x::AbstractArray,
                              cache::VAEDecodeFeatureCache)
    residual = block.shortcut === nothing ? x : block.shortcut(x)
    hidden = _vae_cached_conv3d(
        block.conv1, vae_silu(vae_rmsnorm(block.norm1, x)), cache)
    hidden = _vae_cached_conv3d(
        block.conv2, vae_silu(vae_rmsnorm(block.norm2, hidden)), cache)
    hidden .+ residual
end

function vae_upsample_forward(layer::VAEUpsample, x::AbstractArray)
    if layer.temporal !== nothing
        first_frame = view(x, :, 1:1, :, :, :)
        if size(x, 2) > 1
            expanded_channels = layer.temporal(
                view(x, :, 2:size(x, 2), :, :, :))
            channels = size(expanded_channels, 1) ÷ 2
            temporal = similar(expanded_channels, eltype(expanded_channels),
                channels, 2size(expanded_channels, 2),
                size(expanded_channels, 3), size(expanded_channels, 4),
                size(expanded_channels, 5))
            temporal[:, 1:2:end, :, :, :] .=
                view(expanded_channels, 1:channels, :, :, :, :)
            temporal[:, 2:2:end, :, :, :] .=
                view(expanded_channels, channels + 1:2channels, :, :, :, :)
            x = cat(first_frame, temporal; dims=2)
        else
            x = copy(first_frame)
        end
    end
    upsampled = repeat(x; inner=(1, 1, 2, 2, 1))
    vae_conv3d(layer.spatial, upsampled;
        temporal_left=0, spatial_padding=(1, 1, 1, 1))
end

function vae_upsample_forward(layer::VAEUpsample, x::AbstractArray,
                              cache::VAEDecodeFeatureCache)
    if layer.temporal !== nothing
        index, previous = _next_vae_cache!(cache)
        if previous === nothing
            cache.values[index] = :rep
        else
            cache.values[index] = _vae_cache_tail(x, previous)
            expanded_channels = if previous === :rep
                layer.temporal(x)
            else
                input = cat(previous, x; dims=2)
                temporal_left =
                    max(0, 2 * layer.temporal.padding[1] - size(previous, 2))
                vae_conv3d(
                    layer.temporal, input; temporal_left=temporal_left)
            end
            channels = size(expanded_channels, 1) ÷ 2
            temporal = similar(expanded_channels, eltype(expanded_channels),
                channels, 2size(expanded_channels, 2),
                size(expanded_channels, 3), size(expanded_channels, 4),
                size(expanded_channels, 5))
            temporal[:, 1:2:end, :, :, :] .=
                view(expanded_channels, 1:channels, :, :, :, :)
            temporal[:, 2:2:end, :, :, :] .=
                view(expanded_channels, channels + 1:2channels, :, :, :, :)
            x = temporal
        end
    end
    upsampled = repeat(x; inner=(1, 1, 2, 2, 1))
    vae_conv3d(layer.spatial, upsampled;
        temporal_left=0, spatial_padding=(1, 1, 1, 1))
end

function vae_downsample_forward(layer::VAEDownsample, x::AbstractArray)
    spatial = vae_conv3d(layer.spatial, x;
        temporal_left=0, spatial_padding=(0, 1, 0, 1))
    layer.temporal === nothing && return spatial
    first_frame = view(spatial, :, 1:1, :, :, :)
    size(spatial, 2) == 1 && return copy(first_frame)
    downsampled = vae_conv3d(layer.temporal, spatial;
        temporal_left=0, temporal_right=0)
    cat(first_frame, downsampled; dims=2)
end

struct WanVAEConfig
    base_channels::Int
    latent_channels::Int
    channel_multipliers::Vector{Int}
    residual_blocks::Int
    temporal_downsample::Vector{Bool}
end

function WanVAEConfig(; base_channels=96, latent_channels=16,
                      channel_multipliers=[1, 2, 4, 4],
                      residual_blocks=2,
                      temporal_downsample=[false, true, true])
    length(temporal_downsample) == length(channel_multipliers) - 1 ||
        throw(ArgumentError("temporal_downsample must describe each downsample"))
    WanVAEConfig(base_channels, latent_channels,
        Int.(channel_multipliers), residual_blocks,
        Bool.(temporal_downsample))
end

const VAEEncoderLayer =
    Union{VAEResidualBlock,VAEAttentionBlock,VAEDownsample}

struct WanVAEEncoder
    config::WanVAEConfig
    input_conv::VAEConv3D
    downsample_layers::Vector{VAEEncoderLayer}
    middle1::VAEResidualBlock
    middle_attention::VAEAttentionBlock
    middle2::VAEResidualBlock
    head_norm::VAERMSNorm
    head_conv::VAEConv3D
    quant_conv::VAEConv3D
end

const VAEDecoderLayer =
    Union{VAEResidualBlock,VAEAttentionBlock,VAEUpsample}

struct WanVAEDecoder
    config::WanVAEConfig
    post_quant_conv::VAEConv3D
    input_conv::VAEConv3D
    middle1::VAEResidualBlock
    middle_attention::VAEAttentionBlock
    middle2::VAEResidualBlock
    upsample_layers::Vector{VAEDecoderLayer}
    head_norm::VAERMSNorm
    head_conv::VAEConv3D
end

struct WanVAE
    encoder::WanVAEEncoder
    decoder::WanVAEDecoder
end

function WanVAEEncoder(config::WanVAEConfig=WanVAEConfig();
                       rng=Random.default_rng(), initialize=true)
    conv(output, input, kernel; stride=(1, 1, 1), padding=ntuple(_ -> kernel ÷ 2, 3),
         zero=false) = VAEConv3D(
        (zero || !initialize) ? zeros(Float32, output, input, kernel, kernel, kernel) :
            randn(rng, Float32, output, input, kernel, kernel, kernel) ./
                sqrt(Float32(input * kernel^3)),
        zeros(Float32, output); stride=stride, padding=padding)
    conv2d(output, input; stride=(1, 1, 1)) = VAEConv3D(
        initialize ? randn(rng, Float32, output, input, 1, 3, 3) ./
            sqrt(Float32(input * 9)) : zeros(Float32, output, input, 1, 3, 3),
        zeros(Float32, output); stride=stride)
    residual(input, output) = VAEResidualBlock(
        VAERMSNorm(ones(Float32, input)), conv(output, input, 3),
        VAERMSNorm(ones(Float32, output)), conv(output, output, 3),
        input == output ? nothing : conv(output, input, 1; padding=(0, 0, 0)))
    dimensions = config.base_channels .* vcat(1, config.channel_multipliers)
    layers = VAEEncoderLayer[]
    layer_index = 0
    for stage in eachindex(config.channel_multipliers)
        input_channels = dimensions[stage]
        output_channels = dimensions[stage + 1]
        for _ in 1:config.residual_blocks
            push!(layers, residual(input_channels, output_channels))
            input_channels = output_channels
            layer_index += 1
        end
        if stage < length(config.channel_multipliers)
            spatial = conv2d(output_channels, output_channels;
                stride=(1, 2, 2))
            temporal = config.temporal_downsample[stage] ?
                VAEConv3D(initialize ?
                    randn(rng, Float32, output_channels, output_channels, 3, 1, 1) ./
                        sqrt(Float32(output_channels * 3)) :
                    zeros(Float32, output_channels, output_channels, 3, 1, 1),
                    zeros(Float32, output_channels); stride=(2, 1, 1)) :
                nothing
            push!(layers, VAEDownsample(spatial, temporal))
            layer_index += 1
        end
    end
    final_channels = last(dimensions)
    middle1 = residual(final_channels, final_channels)
    attention = VAEAttentionBlock(VAERMSNorm(ones(Float32, final_channels)),
        conv(3final_channels, final_channels, 1; padding=(0, 0, 0)),
        conv(final_channels, final_channels, 1; padding=(0, 0, 0), zero=true))
    middle2 = residual(final_channels, final_channels)
    head = conv(2config.latent_channels, final_channels, 3)
    quant = conv(2config.latent_channels, 2config.latent_channels, 1;
        padding=(0, 0, 0))
    WanVAEEncoder(config, conv(config.base_channels, 3, 3), layers,
        middle1, attention, middle2, VAERMSNorm(ones(Float32, final_channels)),
        head, quant)
end

function WanVAEDecoder(config::WanVAEConfig=WanVAEConfig();
                       rng=Random.default_rng(), initialize=true)
    conv(output, input, kernel;
         stride=(1, 1, 1), padding=ntuple(_ -> kernel ÷ 2, 3),
         zero=false) = VAEConv3D(
        (zero || !initialize) ?
            zeros(Float32, output, input, kernel, kernel, kernel) :
            randn(rng, Float32, output, input, kernel, kernel, kernel) ./
                sqrt(Float32(input * kernel^3)),
        zeros(Float32, output); stride=stride, padding=padding)
    conv2d(output, input) = VAEConv3D(
        initialize ? randn(rng, Float32, output, input, 1, 3, 3) ./
            sqrt(Float32(input * 9)) :
            zeros(Float32, output, input, 1, 3, 3),
        zeros(Float32, output))
    residual(input, output) = VAEResidualBlock(
        VAERMSNorm(ones(Float32, input)), conv(output, input, 3),
        VAERMSNorm(ones(Float32, output)), conv(output, output, 3),
        input == output ? nothing :
            conv(output, input, 1; padding=(0, 0, 0)))
    dimensions = config.base_channels .*
        vcat(last(config.channel_multipliers),
            reverse(config.channel_multipliers))
    channels = first(dimensions)
    middle1 = residual(channels, channels)
    attention = VAEAttentionBlock(VAERMSNorm(ones(Float32, channels)),
        conv(3channels, channels, 1; padding=(0, 0, 0)),
        conv(channels, channels, 1; padding=(0, 0, 0), zero=true))
    middle2 = residual(channels, channels)
    temporal_upsample = reverse(config.temporal_downsample)
    layers = VAEDecoderLayer[]
    for stage in 1:length(config.channel_multipliers)
        input_channels = dimensions[stage]
        output_channels = dimensions[stage + 1]
        stage > 1 && (input_channels ÷= 2)
        for _ in 1:config.residual_blocks + 1
            push!(layers, residual(input_channels, output_channels))
            input_channels = output_channels
        end
        if stage < length(config.channel_multipliers)
            temporal = temporal_upsample[stage] ?
                VAEConv3D(initialize ?
                    randn(rng, Float32, 2output_channels, output_channels,
                        3, 1, 1) ./ sqrt(Float32(output_channels * 3)) :
                    zeros(Float32, 2output_channels, output_channels, 3, 1, 1),
                    zeros(Float32, 2output_channels); padding=(1, 0, 0)) :
                nothing
            push!(layers, VAEUpsample(
                conv2d(output_channels ÷ 2, output_channels), temporal))
        end
    end
    final_channels = last(dimensions)
    WanVAEDecoder(config,
        conv(config.latent_channels, config.latent_channels, 1;
            padding=(0, 0, 0)),
        conv(first(dimensions), config.latent_channels, 3),
        middle1, attention, middle2, layers,
        VAERMSNorm(ones(Float32, final_channels)),
        conv(3, final_channels, 3))
end

function wan_vae_encoder_forward(model::WanVAEEncoder,
                                 video::AbstractArray; scale=true)
    ndims(video) == 5 ||
        throw(DimensionMismatch("Wan VAE input must be (channels,frames,height,width,batch)"))
    size(video, 1) == 3 ||
        throw(DimensionMismatch("Wan VAE input must have three RGB channels"))
    x = model.input_conv(video)
    for layer in model.downsample_layers
        x = layer isa VAEResidualBlock ? vae_residual_forward(layer, x) :
            layer isa VAEAttentionBlock ? vae_attention_forward(layer, x) :
            vae_downsample_forward(layer, x)
    end
    x = vae_residual_forward(model.middle1, x)
    x = vae_attention_forward(model.middle_attention, x)
    x = vae_residual_forward(model.middle2, x)
    x = model.head_conv(vae_silu(vae_rmsnorm(model.head_norm, x)))
    moments = model.quant_conv(x)
    mu = moments[1:model.config.latent_channels, :, :, :, :]
    scale ? scale_wan_latents(mu) : mu
end

function _wan_vae_decoder_full_forward(model::WanVAEDecoder, z::AbstractArray)
    x = model.input_conv(model.post_quant_conv(z))
    x = vae_residual_forward(model.middle1, x)
    x = vae_attention_forward(model.middle_attention, x)
    x = vae_residual_forward(model.middle2, x)
    for layer in model.upsample_layers
        x = layer isa VAEResidualBlock ? vae_residual_forward(layer, x) :
            layer isa VAEAttentionBlock ? vae_attention_forward(layer, x) :
            vae_upsample_forward(layer, x)
    end
    model.head_conv(vae_silu(vae_rmsnorm(model.head_norm, x)))
end

function _wan_vae_decoder_streaming_forward(
        model::WanVAEDecoder, z::AbstractArray)
    # Wan's reference decoder processes one latent frame at a time and carries
    # the last two inputs for each causal convolution. Besides matching the
    # reference execution order, this bounds full-resolution Conv3D tensors.
    x = model.post_quant_conv(z)
    cache = VAEDecodeFeatureCache()
    outputs = Any[]
    for frame in axes(x, 2)
        cache.index = 0
        chunk = view(x, :, frame:frame, :, :, :)
        chunk = _vae_cached_conv3d(model.input_conv, chunk, cache)
        chunk = vae_residual_forward(model.middle1, chunk, cache)
        chunk = vae_attention_forward(model.middle_attention, chunk)
        chunk = vae_residual_forward(model.middle2, chunk, cache)
        for layer in model.upsample_layers
            chunk = layer isa VAEResidualBlock ?
                vae_residual_forward(layer, chunk, cache) :
                layer isa VAEAttentionBlock ?
                vae_attention_forward(layer, chunk) :
                vae_upsample_forward(layer, chunk, cache)
        end
        chunk = _vae_cached_conv3d(
            model.head_conv,
            vae_silu(vae_rmsnorm(model.head_norm, chunk)), cache)
        push!(outputs, chunk)
    end
    cat(outputs...; dims=2)
end

function wan_vae_decoder_forward(model::WanVAEDecoder,
                                 latents::AbstractArray; unscale=true,
                                 clamp=true, streaming=true)
    ndims(latents) == 5 ||
        throw(DimensionMismatch("Wan VAE latents must be (channels,frames,height,width,batch)"))
    size(latents, 1) == model.config.latent_channels ||
        throw(DimensionMismatch("Wan VAE latent channel count differs"))
    z = unscale ? unscale_wan_latents(latents) : latents
    decoded = streaming ?
        _wan_vae_decoder_streaming_forward(model, z) :
        _wan_vae_decoder_full_forward(model, z)
    clamp ? Base.clamp.(decoded, -1f0, 1f0) : decoded
end

function _vae_conv_specs!(specs, prefix, output, input, kernel::NTuple{3,Int};
                          conv2d=false)
    shape = conv2d ?
        [output, input, kernel[2], kernel[3]] :
        [output, input, kernel...]
    push!(specs, TensorSpec("$prefix.weight", shape, CONV_OUT_IN_SPATIAL))
    push!(specs, TensorSpec("$prefix.bias", [output], VECTOR_LAYOUT))
end

function _vae_norm_spec!(specs, prefix, channels; images=false)
    shape = images ? [channels, 1, 1] : [channels, 1, 1, 1]
    push!(specs, TensorSpec("$prefix.gamma", shape, ROW_MAJOR_SOURCE))
end

function _vae_residual_specs!(specs, prefix, input, output)
    _vae_norm_spec!(specs, "$prefix.residual.0", input)
    _vae_conv_specs!(specs, "$prefix.residual.2", output, input, (3, 3, 3))
    _vae_norm_spec!(specs, "$prefix.residual.3", output)
    _vae_conv_specs!(specs, "$prefix.residual.6", output, output, (3, 3, 3))
    input == output ||
        _vae_conv_specs!(specs, "$prefix.shortcut", output, input, (1, 1, 1))
end

function _vae_attention_specs!(specs, prefix, channels)
    _vae_norm_spec!(specs, "$prefix.norm", channels; images=true)
    _vae_conv_specs!(specs, "$prefix.to_qkv", 3channels, channels, (1, 1, 1);
        conv2d=true)
    _vae_conv_specs!(specs, "$prefix.proj", channels, channels, (1, 1, 1);
        conv2d=true)
end

function wan_vae_encoder_specs(config::WanVAEConfig=WanVAEConfig())
    specs = TensorSpec[]
    _vae_conv_specs!(specs, "encoder.conv1", config.base_channels, 3, (3, 3, 3))
    dimensions = config.base_channels .* vcat(1, config.channel_multipliers)
    layer_index = 0
    for stage in eachindex(config.channel_multipliers)
        input_channels = dimensions[stage]
        output_channels = dimensions[stage + 1]
        for _ in 1:config.residual_blocks
            _vae_residual_specs!(specs, "encoder.downsamples.$layer_index",
                input_channels, output_channels)
            input_channels = output_channels
            layer_index += 1
        end
        if stage < length(config.channel_multipliers)
            prefix = "encoder.downsamples.$layer_index"
            _vae_conv_specs!(specs, "$prefix.resample.1",
                output_channels, output_channels, (1, 3, 3); conv2d=true)
            config.temporal_downsample[stage] &&
                _vae_conv_specs!(specs, "$prefix.time_conv",
                    output_channels, output_channels, (3, 1, 1))
            layer_index += 1
        end
    end
    final_channels = last(dimensions)
    _vae_residual_specs!(specs, "encoder.middle.0",
        final_channels, final_channels)
    _vae_attention_specs!(specs, "encoder.middle.1", final_channels)
    _vae_residual_specs!(specs, "encoder.middle.2",
        final_channels, final_channels)
    _vae_norm_spec!(specs, "encoder.head.0", final_channels)
    _vae_conv_specs!(specs, "encoder.head.2", 2config.latent_channels,
        final_channels, (3, 3, 3))
    _vae_conv_specs!(specs, "conv1", 2config.latent_channels,
        2config.latent_channels, (1, 1, 1))
    specs
end

function wan_vae_decoder_specs(config::WanVAEConfig=WanVAEConfig())
    specs = TensorSpec[]
    _vae_conv_specs!(specs, "conv2", config.latent_channels,
        config.latent_channels, (1, 1, 1))
    dimensions = config.base_channels .*
        vcat(last(config.channel_multipliers),
            reverse(config.channel_multipliers))
    first_channels = first(dimensions)
    _vae_conv_specs!(specs, "decoder.conv1", first_channels,
        config.latent_channels, (3, 3, 3))
    _vae_residual_specs!(specs, "decoder.middle.0",
        first_channels, first_channels)
    _vae_attention_specs!(specs, "decoder.middle.1", first_channels)
    _vae_residual_specs!(specs, "decoder.middle.2",
        first_channels, first_channels)
    temporal_upsample = reverse(config.temporal_downsample)
    layer_index = 0
    for stage in 1:length(config.channel_multipliers)
        input_channels = dimensions[stage]
        output_channels = dimensions[stage + 1]
        stage > 1 && (input_channels ÷= 2)
        for _ in 1:config.residual_blocks + 1
            _vae_residual_specs!(specs, "decoder.upsamples.$layer_index",
                input_channels, output_channels)
            input_channels = output_channels
            layer_index += 1
        end
        if stage < length(config.channel_multipliers)
            prefix = "decoder.upsamples.$layer_index"
            _vae_conv_specs!(specs, "$prefix.resample.1",
                output_channels ÷ 2, output_channels, (1, 3, 3); conv2d=true)
            temporal_upsample[stage] &&
                _vae_conv_specs!(specs, "$prefix.time_conv",
                    2output_channels, output_channels, (3, 1, 1))
            layer_index += 1
        end
    end
    final_channels = last(dimensions)
    _vae_norm_spec!(specs, "decoder.head.0", final_channels)
    _vae_conv_specs!(specs, "decoder.head.2", 3, final_channels, (3, 3, 3))
    specs
end

wan_vae_specs(config::WanVAEConfig=WanVAEConfig()) =
    vcat(wan_vae_encoder_specs(config), wan_vae_decoder_specs(config))

function _wan_vae_diffusers_residual_suffix(suffix::AbstractString)
    replacements = (
        "residual.0.gamma" => "norm1.gamma",
        "residual.2.bias" => "conv1.bias",
        "residual.2.weight" => "conv1.weight",
        "residual.3.gamma" => "norm2.gamma",
        "residual.6.bias" => "conv2.bias",
        "residual.6.weight" => "conv2.weight",
        "shortcut.bias" => "conv_shortcut.bias",
        "shortcut.weight" => "conv_shortcut.weight",
    )
    for (original, diffusers) in replacements
        suffix == original && return diffusers
    end
    String(suffix)
end

function _wan_vae_diffusers_middle_key(key::AbstractString)
    parts = split(key, '.')
    length(parts) >= 4 || return String(key)
    side = parts[1]
    block = parse(Int, parts[3])
    suffix = join(parts[4:end], ".")
    if block == 0 || block == 2
        resnet = block ÷ 2
        converted = _wan_vae_diffusers_residual_suffix(suffix)
        return "$side.mid_block.resnets.$resnet.$converted"
    elseif block == 1
        return "$side.mid_block.attentions.0.$suffix"
    end
    String(key)
end

"""
Translate one original Wan VAE state-dictionary key to the official Diffusers
`AutoencoderKLWan` SafeTensors key.
"""
function wan_vae_diffusers_key(key::AbstractString)
    key == "encoder.conv1.weight" && return "encoder.conv_in.weight"
    key == "encoder.conv1.bias" && return "encoder.conv_in.bias"
    key == "decoder.conv1.weight" && return "decoder.conv_in.weight"
    key == "decoder.conv1.bias" && return "decoder.conv_in.bias"
    key == "conv1.weight" && return "quant_conv.weight"
    key == "conv1.bias" && return "quant_conv.bias"
    key == "conv2.weight" && return "post_quant_conv.weight"
    key == "conv2.bias" && return "post_quant_conv.bias"

    startswith(key, "encoder.middle.") &&
        return _wan_vae_diffusers_middle_key(key)
    startswith(key, "decoder.middle.") &&
        return _wan_vae_diffusers_middle_key(key)

    head_mappings = (
        "encoder.head.0.gamma" => "encoder.norm_out.gamma",
        "encoder.head.2.weight" => "encoder.conv_out.weight",
        "encoder.head.2.bias" => "encoder.conv_out.bias",
        "decoder.head.0.gamma" => "decoder.norm_out.gamma",
        "decoder.head.2.weight" => "decoder.conv_out.weight",
        "decoder.head.2.bias" => "decoder.conv_out.bias",
    )
    for (original, diffusers) in head_mappings
        key == original && return diffusers
    end

    if startswith(key, "encoder.downsamples.")
        parts = split(key, '.')
        length(parts) >= 4 || return String(key)
        block = parts[3]
        suffix = join(parts[4:end], ".")
        converted = _wan_vae_diffusers_residual_suffix(suffix)
        return "encoder.down_blocks.$block.$converted"
    end

    if startswith(key, "decoder.upsamples.")
        parts = split(key, '.')
        length(parts) >= 4 || return String(key)
        block = parse(Int, parts[3])
        suffix = join(parts[4:end], ".")
        group = block ÷ 4
        within_group = block % 4
        if startswith(suffix, "residual.")
            within_group <= 2 || return String(key)
            converted = _wan_vae_diffusers_residual_suffix(suffix)
            return "decoder.up_blocks.$group.resnets.$within_group.$converted"
        elseif startswith(suffix, "shortcut.")
            within_group <= 2 || return String(key)
            converted = _wan_vae_diffusers_residual_suffix(suffix)
            return "decoder.up_blocks.$group.resnets.$within_group.$converted"
        elseif (startswith(suffix, "resample.") ||
                startswith(suffix, "time_conv.")) && within_group == 3
            return "decoder.up_blocks.$group.upsamplers.0.$suffix"
        end
        return "decoder.up_blocks.$block.$suffix"
    end

    String(key)
end

function wan_vae_diffusers_specs(config::WanVAEConfig=WanVAEConfig())
    [TensorSpec(wan_vae_diffusers_key(spec.source_key), spec.source_shape,
                spec.layout; destination=spec.destination_key,
                required=spec.required)
     for spec in wan_vae_specs(config)]
end

function _wan_vae_tensor_source(source::AbstractTensorSource,
                                config::WanVAEConfig)
    keys = Set(tensor_keys(source))
    native = "encoder.conv1.weight" in keys ||
             "decoder.conv1.weight" in keys
    diffusers = "encoder.conv_in.weight" in keys ||
                "decoder.conv_in.weight" in keys
    native && diffusers &&
        throw(ArgumentError("Wan VAE checkpoint mixes original and Diffusers keys"))
    diffusers || return source

    aliases = Dict{String,String}()
    mapped_source_keys = Set{String}()
    for spec in wan_vae_specs(config)
        source_key = wan_vae_diffusers_key(spec.source_key)
        push!(mapped_source_keys, source_key)
        source_key in keys &&
            (aliases[spec.source_key] = source_key)
    end
    for key in keys
        key in mapped_source_keys && continue
        haskey(aliases, key) &&
            throw(ArgumentError("Wan VAE key mapping collides at $key"))
        aliases[key] = key
    end
    KeyMappedTensorSource(source, aliases)
end

function _load_vae_conv!(source, prefix, layer::VAEConv3D; conv2d=false)
    official_shape = conv2d ?
        [size(layer.weight, 1), size(layer.weight, 2),
         size(layer.weight, 4), size(layer.weight, 5)] :
        collect(size(layer.weight))
    loaded = load_state_tensor(source,
        TensorSpec("$prefix.weight", official_shape, CONV_OUT_IN_SPATIAL))
    if ndims(loaded) == 4
        loaded = reshape(loaded, size(loaded, 1), size(loaded, 2), 1,
            size(loaded, 3), size(loaded, 4))
    end
    copyto!(layer.weight, _umt5_compute_array(loaded))
    copyto!(layer.bias, _umt5_compute_array(load_state_tensor(source,
        TensorSpec("$prefix.bias", [length(layer.bias)], VECTOR_LAYOUT))))
    layer
end

function _load_vae_norm!(source, prefix, norm::VAERMSNorm; images=false)
    shape = images ? [length(norm.gamma), 1, 1] :
        [length(norm.gamma), 1, 1, 1]
    loaded = load_state_tensor(source,
        TensorSpec("$prefix.gamma", shape, ROW_MAJOR_SOURCE))
    copyto!(norm.gamma, vec(_umt5_compute_array(loaded)))
    norm
end

function _load_vae_residual!(source, prefix, block::VAEResidualBlock)
    _load_vae_norm!(source, "$prefix.residual.0", block.norm1)
    _load_vae_conv!(source, "$prefix.residual.2", block.conv1)
    _load_vae_norm!(source, "$prefix.residual.3", block.norm2)
    _load_vae_conv!(source, "$prefix.residual.6", block.conv2)
    block.shortcut === nothing ||
        _load_vae_conv!(source, "$prefix.shortcut", block.shortcut)
    block
end

function _load_vae_attention!(source, prefix, block::VAEAttentionBlock)
    _load_vae_norm!(source, "$prefix.norm", block.norm; images=true)
    _load_vae_conv!(source, "$prefix.to_qkv", block.qkv; conv2d=true)
    _load_vae_conv!(source, "$prefix.proj", block.projection; conv2d=true)
    block
end

function load_wan_vae_encoder(source::AbstractTensorSource,
                              config::WanVAEConfig=WanVAEConfig())
    source = _wan_vae_tensor_source(source, config)
    audit = audit_state_dict(source, wan_vae_encoder_specs(config);
        allow_unexpected=true)
    isempty(audit.missing) && isempty(audit.shape_mismatches) ||
        _assert_clean_audit(StateDictAudit(audit.missing, String[],
            audit.shape_mismatches))
    model = WanVAEEncoder(config; rng=Xoshiro(0), initialize=false)
    _load_vae_conv!(source, "encoder.conv1", model.input_conv)
    layer_index = 0
    for layer in model.downsample_layers
        prefix = "encoder.downsamples.$layer_index"
        if layer isa VAEResidualBlock
            _load_vae_residual!(source, prefix, layer)
        else
            _load_vae_conv!(source, "$prefix.resample.1", layer.spatial;
                conv2d=true)
            layer.temporal === nothing ||
                _load_vae_conv!(source, "$prefix.time_conv", layer.temporal)
        end
        layer_index += 1
    end
    _load_vae_residual!(source, "encoder.middle.0", model.middle1)
    _load_vae_attention!(source, "encoder.middle.1", model.middle_attention)
    _load_vae_residual!(source, "encoder.middle.2", model.middle2)
    _load_vae_norm!(source, "encoder.head.0", model.head_norm)
    _load_vae_conv!(source, "encoder.head.2", model.head_conv)
    _load_vae_conv!(source, "conv1", model.quant_conv)
    model
end

load_wan_vae_encoder(path::AbstractString,
                     config::WanVAEConfig=WanVAEConfig()) =
    load_wan_vae_encoder(open_tensor_source(path), config)

function load_wan_vae_decoder(source::AbstractTensorSource,
                              config::WanVAEConfig=WanVAEConfig())
    source = _wan_vae_tensor_source(source, config)
    audit = audit_state_dict(source, wan_vae_decoder_specs(config);
        allow_unexpected=true)
    isempty(audit.missing) && isempty(audit.shape_mismatches) ||
        _assert_clean_audit(StateDictAudit(audit.missing, String[],
            audit.shape_mismatches))
    model = WanVAEDecoder(config; rng=Xoshiro(0), initialize=false)
    _load_vae_conv!(source, "conv2", model.post_quant_conv)
    _load_vae_conv!(source, "decoder.conv1", model.input_conv)
    _load_vae_residual!(source, "decoder.middle.0", model.middle1)
    _load_vae_attention!(source, "decoder.middle.1", model.middle_attention)
    _load_vae_residual!(source, "decoder.middle.2", model.middle2)
    for (offset, layer) in enumerate(model.upsample_layers)
        prefix = "decoder.upsamples.$(offset - 1)"
        if layer isa VAEResidualBlock
            _load_vae_residual!(source, prefix, layer)
        else
            _load_vae_conv!(source, "$prefix.resample.1", layer.spatial;
                conv2d=true)
            layer.temporal === nothing ||
                _load_vae_conv!(source, "$prefix.time_conv", layer.temporal)
        end
    end
    _load_vae_norm!(source, "decoder.head.0", model.head_norm)
    _load_vae_conv!(source, "decoder.head.2", model.head_conv)
    model
end

load_wan_vae_decoder(path::AbstractString,
                     config::WanVAEConfig=WanVAEConfig()) =
    load_wan_vae_decoder(open_tensor_source(path), config)

function load_wan_vae(source::AbstractTensorSource,
                      config::WanVAEConfig=WanVAEConfig(); strict=true)
    source = _wan_vae_tensor_source(source, config)
    audit = audit_state_dict(source, wan_vae_specs(config);
        allow_unexpected=!strict)
    _assert_clean_audit(audit)
    WanVAE(load_wan_vae_encoder(source, config),
        load_wan_vae_decoder(source, config))
end

load_wan_vae(path::AbstractString,
             config::WanVAEConfig=WanVAEConfig(); kwargs...) =
    load_wan_vae(open_tensor_source(path), config; kwargs...)

function _store_vae_conv!(state, prefix, layer::VAEConv3D; conv2d=false)
    weight = Array(layer.weight)
    state["$prefix.weight"] = conv2d ? dropdims(weight; dims=3) : weight
    state["$prefix.bias"] = Array(layer.bias)
end

function _store_vae_norm!(state, prefix, norm::VAERMSNorm; images=false)
    trailing = images ? (1, 1) : (1, 1, 1)
    state["$prefix.gamma"] =
        reshape(Array(norm.gamma), length(norm.gamma), trailing...)
end

function _store_vae_residual!(state, prefix, block::VAEResidualBlock)
    _store_vae_norm!(state, "$prefix.residual.0", block.norm1)
    _store_vae_conv!(state, "$prefix.residual.2", block.conv1)
    _store_vae_norm!(state, "$prefix.residual.3", block.norm2)
    _store_vae_conv!(state, "$prefix.residual.6", block.conv2)
    block.shortcut === nothing ||
        _store_vae_conv!(state, "$prefix.shortcut", block.shortcut)
end

function _store_vae_attention!(state, prefix, block::VAEAttentionBlock)
    _store_vae_norm!(state, "$prefix.norm", block.norm; images=true)
    _store_vae_conv!(state, "$prefix.to_qkv", block.qkv; conv2d=true)
    _store_vae_conv!(state, "$prefix.proj", block.projection; conv2d=true)
end

function wan_vae_encoder_state_dict(model::WanVAEEncoder)
    state = Dict{String,AbstractArray}()
    _store_vae_conv!(state, "encoder.conv1", model.input_conv)
    for (offset, layer) in enumerate(model.downsample_layers)
        prefix = "encoder.downsamples.$(offset - 1)"
        if layer isa VAEResidualBlock
            _store_vae_residual!(state, prefix, layer)
        else
            _store_vae_conv!(state, "$prefix.resample.1", layer.spatial;
                conv2d=true)
            layer.temporal === nothing ||
                _store_vae_conv!(state, "$prefix.time_conv", layer.temporal)
        end
    end
    _store_vae_residual!(state, "encoder.middle.0", model.middle1)
    _store_vae_attention!(state, "encoder.middle.1", model.middle_attention)
    _store_vae_residual!(state, "encoder.middle.2", model.middle2)
    _store_vae_norm!(state, "encoder.head.0", model.head_norm)
    _store_vae_conv!(state, "encoder.head.2", model.head_conv)
    _store_vae_conv!(state, "conv1", model.quant_conv)
    state
end

function wan_vae_decoder_state_dict(model::WanVAEDecoder)
    state = Dict{String,AbstractArray}()
    _store_vae_conv!(state, "conv2", model.post_quant_conv)
    _store_vae_conv!(state, "decoder.conv1", model.input_conv)
    _store_vae_residual!(state, "decoder.middle.0", model.middle1)
    _store_vae_attention!(state, "decoder.middle.1", model.middle_attention)
    _store_vae_residual!(state, "decoder.middle.2", model.middle2)
    for (offset, layer) in enumerate(model.upsample_layers)
        prefix = "decoder.upsamples.$(offset - 1)"
        if layer isa VAEResidualBlock
            _store_vae_residual!(state, prefix, layer)
        else
            _store_vae_conv!(state, "$prefix.resample.1", layer.spatial;
                conv2d=true)
            layer.temporal === nothing ||
                _store_vae_conv!(state, "$prefix.time_conv", layer.temporal)
        end
    end
    _store_vae_norm!(state, "decoder.head.0", model.head_norm)
    _store_vae_conv!(state, "decoder.head.2", model.head_conv)
    state
end

_move_vae_conv(layer::VAEConv3D, transfer) =
    VAEConv3D(transfer(layer.weight), transfer(layer.bias);
        stride=layer.stride, padding=layer.padding)
_move_vae_norm(norm::VAERMSNorm, transfer) = VAERMSNorm(transfer(norm.gamma))
function _move_vae_residual(block::VAEResidualBlock, transfer)
    VAEResidualBlock(_move_vae_norm(block.norm1, transfer),
        _move_vae_conv(block.conv1, transfer),
        _move_vae_norm(block.norm2, transfer),
        _move_vae_conv(block.conv2, transfer),
        block.shortcut === nothing ? nothing :
            _move_vae_conv(block.shortcut, transfer))
end
_move_vae_attention(block::VAEAttentionBlock, transfer) =
    VAEAttentionBlock(_move_vae_norm(block.norm, transfer),
        _move_vae_conv(block.qkv, transfer),
        _move_vae_conv(block.projection, transfer))
_move_vae_downsample(layer::VAEDownsample, transfer) =
    VAEDownsample(_move_vae_conv(layer.spatial, transfer),
        layer.temporal === nothing ? nothing :
            _move_vae_conv(layer.temporal, transfer))

function move_to_device(model::WanVAEEncoder, transfer)
    layers = VAEEncoderLayer[
        layer isa VAEResidualBlock ? _move_vae_residual(layer, transfer) :
        layer isa VAEAttentionBlock ? _move_vae_attention(layer, transfer) :
        _move_vae_downsample(layer, transfer)
        for layer in model.downsample_layers
    ]
    WanVAEEncoder(model.config, _move_vae_conv(model.input_conv, transfer),
        layers, _move_vae_residual(model.middle1, transfer),
        _move_vae_attention(model.middle_attention, transfer),
        _move_vae_residual(model.middle2, transfer),
        _move_vae_norm(model.head_norm, transfer),
        _move_vae_conv(model.head_conv, transfer),
        _move_vae_conv(model.quant_conv, transfer))
end

function move_to_device(model::WanVAEDecoder, transfer)
    layers = VAEDecoderLayer[
        layer isa VAEResidualBlock ? _move_vae_residual(layer, transfer) :
        layer isa VAEAttentionBlock ? _move_vae_attention(layer, transfer) :
        VAEUpsample(_move_vae_conv(layer.spatial, transfer),
            layer.temporal === nothing ? nothing :
                _move_vae_conv(layer.temporal, transfer))
        for layer in model.upsample_layers
    ]
    WanVAEDecoder(model.config,
        _move_vae_conv(model.post_quant_conv, transfer),
        _move_vae_conv(model.input_conv, transfer),
        _move_vae_residual(model.middle1, transfer),
        _move_vae_attention(model.middle_attention, transfer),
        _move_vae_residual(model.middle2, transfer), layers,
        _move_vae_norm(model.head_norm, transfer),
        _move_vae_conv(model.head_conv, transfer))
end

move_to_device(model::WanVAE, transfer) =
    WanVAE(move_to_device(model.encoder, transfer),
        move_to_device(model.decoder, transfer))

move_to_device(model::WanVAEEncoder, ::Val{:cpu}) =
    move_to_device(model, Array)
function move_to_device(model::WanVAEEncoder, ::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    move_to_device(model, CUDA.CuArray)
end
move_to_device(model::WanVAEEncoder, device::Symbol) =
    device === :cpu ? move_to_device(model, Val(:cpu)) :
    device === :cuda ? move_to_device(model, Val(:cuda)) :
    throw(ArgumentError("unsupported device: $device"))

move_to_device(model::WanVAEDecoder, ::Val{:cpu}) =
    move_to_device(model, Array)
function move_to_device(model::WanVAEDecoder, ::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    move_to_device(model, CUDA.CuArray)
end
move_to_device(model::WanVAEDecoder, device::Symbol) =
    device === :cpu ? move_to_device(model, Val(:cpu)) :
    device === :cuda ? move_to_device(model, Val(:cuda)) :
    throw(ArgumentError("unsupported device: $device"))
move_to_device(model::WanVAEDecoder, device::Symbol, precision::Symbol) =
    move_to_device(model, array_transfer(device, precision))

move_to_device(model::WanVAE, ::Val{:cpu}) = move_to_device(model, Array)
function move_to_device(model::WanVAE, ::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    move_to_device(model, CUDA.CuArray)
end
move_to_device(model::WanVAE, device::Symbol) =
    device === :cpu ? move_to_device(model, Val(:cpu)) :
    device === :cuda ? move_to_device(model, Val(:cuda)) :
    throw(ArgumentError("unsupported device: $device"))
move_to_device(model::WanVAE, device::Symbol, precision::Symbol) =
    move_to_device(model, array_transfer(device, precision))

const WAN_VAE_MEAN = Float32[
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
const WAN_VAE_STD = Float32[
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

function _wan_vae_stat_like(latents::AbstractArray, values::AbstractVector)
    result = similar(latents, eltype(latents), length(values))
    copyto!(result, convert(Vector{eltype(latents)}, values))
    result
end

function scale_wan_latents(latents::AbstractArray;
                           mean=WAN_VAE_MEAN, std=WAN_VAE_STD)
    size(latents, 1) == length(mean) == length(std) ||
        throw(DimensionMismatch("Wan latent channel count differs from scale"))
    shape = (:, ntuple(_ -> 1, ndims(latents) - 1)...)
    device_mean = _wan_vae_stat_like(latents, mean)
    device_std = _wan_vae_stat_like(latents, std)
    (latents .- reshape(device_mean, shape...)) ./
        reshape(device_std, shape...)
end

function unscale_wan_latents(latents::AbstractArray;
                             mean=WAN_VAE_MEAN, std=WAN_VAE_STD)
    size(latents, 1) == length(mean) == length(std) ||
        throw(DimensionMismatch("Wan latent channel count differs from scale"))
    shape = (:, ntuple(_ -> 1, ndims(latents) - 1)...)
    device_mean = _wan_vae_stat_like(latents, mean)
    device_std = _wan_vae_stat_like(latents, std)
    latents .* reshape(device_std, shape...) .+
        reshape(device_mean, shape...)
end
