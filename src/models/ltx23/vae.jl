# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

struct LTXVAEBlockConfig
    kind::Symbol
    num_layers::Int
    multiplier::Int
end

const LTX23_VAE_DEFAULT_BLOCKS = LTXVAEBlockConfig[
    LTXVAEBlockConfig(:res_x, 4, 1),
    LTXVAEBlockConfig(:compress_space_res, 0, 2),
    LTXVAEBlockConfig(:res_x, 6, 1),
    LTXVAEBlockConfig(:compress_time_res, 0, 2),
    LTXVAEBlockConfig(:res_x, 4, 1),
    LTXVAEBlockConfig(:compress_all_res, 0, 2),
    LTXVAEBlockConfig(:res_x, 2, 1),
    LTXVAEBlockConfig(:compress_all_res, 0, 1),
    LTXVAEBlockConfig(:res_x, 2, 1),
]

struct LTXVideoVAEConfig
    input_channels::Int
    latent_channels::Int
    patch_size::Int
    blocks::Vector{LTXVAEBlockConfig}
    latent_log_variance::Symbol
    normalization::Symbol
    spatial_padding::Symbol
    epsilon::Float32
end

function LTXVideoVAEConfig(; input_channels=3, latent_channels=128,
                           patch_size=4,
                           blocks=copy(LTX23_VAE_DEFAULT_BLOCKS),
                           latent_log_variance=:uniform,
                           normalization=:pixel_norm,
                           spatial_padding=:zeros, epsilon=1f-8)
    input_channels > 0 && latent_channels > 0 && patch_size > 0 ||
        throw(ArgumentError("LTX VAE channel and patch sizes must be positive"))
    normalization === :pixel_norm ||
        throw(ArgumentError("native LTX VAE currently requires pixel_norm"))
    spatial_padding === :zeros ||
        throw(ArgumentError("native LTX VAE encoder currently requires zero spatial padding"))
    latent_log_variance in (:uniform, :per_channel, :constant, :none) ||
        throw(ArgumentError("unsupported LTX latent log-variance mode"))
    LTXVideoVAEConfig(Int(input_channels), Int(latent_channels),
        Int(patch_size), collect(blocks), Symbol(latent_log_variance),
        Symbol(normalization), Symbol(spatial_padding), Float32(epsilon))
end

function _ltx_vae_block(value)
    value isa AbstractVector && length(value) == 2 ||
        throw(ArgumentError("LTX VAE block must be [name, parameters]"))
    name, parameters = Symbol(value[1]), value[2]
    values = parameters isa Integer ?
        Dict("num_layers" => parameters) : parameters
    values isa AbstractDict ||
        throw(ArgumentError("LTX VAE block parameters must be an object"))
    LTXVAEBlockConfig(name,
        Int(_ltx_config_value(values, "num_layers", 0)),
        Int(_ltx_config_value(values, "multiplier", 1)))
end

function ltx23_vae_config(values::AbstractDict)
    vae = get(values, "vae", values)
    blocks = [_ltx_vae_block(value) for value in
        _ltx_config_value(vae, "encoder_blocks",
                          [[String(block.kind),
                            Dict("num_layers" => block.num_layers,
                                 "multiplier" => block.multiplier)]
                           for block in LTX23_VAE_DEFAULT_BLOCKS])]
    LTXVideoVAEConfig(
        input_channels=Int(_ltx_config_value(vae, "in_channels", 3)),
        latent_channels=Int(_ltx_config_value(vae, "latent_channels", 128)),
        patch_size=Int(_ltx_config_value(vae, "patch_size", 4)),
        blocks=blocks,
        latent_log_variance=Symbol(_ltx_config_value(
            vae, "latent_log_var", "uniform")),
        normalization=Symbol(_ltx_config_value(vae, "norm_layer", "pixel_norm")),
        spatial_padding=Symbol(_ltx_config_value(
            vae, "spatial_padding_mode", "zeros")))
end

function ltx23_vae_config(source::AbstractString)
    values = if endswith(lowercase(source), ".safetensors")
        metadata = inspect_safetensors(source).metadata
        haskey(metadata, "config") ||
            throw(ArgumentError("LTX checkpoint has no `config` metadata"))
        parse_json(metadata["config"])
    else
        parse_json(read(source, String))
    end
    values isa AbstractDict ||
        throw(ArgumentError("LTX configuration root must be a JSON object"))
    ltx23_vae_config(values)
end

function _ltx_vae_conv_specs!(specs, prefix, output, input)
    push!(specs,
        TensorSpec("$prefix.weight", [output, input, 3, 3, 3],
                   CONV_OUT_IN_SPATIAL),
        TensorSpec("$prefix.bias", [output], VECTOR_LAYOUT))
end

"""Exact official LTX-2.3 video encoder checkpoint inventory."""
function ltx23_vae_encoder_specs(config::LTXVideoVAEConfig=LTXVideoVAEConfig();
                                 checkpoint_prefix="vae.encoder.",
                                 statistics_prefix="vae.per_channel_statistics.")
    specs = TensorSpec[]
    channels = config.latent_channels
    _ltx_vae_conv_specs!(specs, checkpoint_prefix * "conv_in.conv",
        channels, config.input_channels * config.patch_size^2)
    for (block_index, block) in enumerate(config.blocks)
        prefix = checkpoint_prefix * "down_blocks.$(block_index - 1)"
        if block.kind === :res_x
            for layer in 0:block.num_layers-1
                base = "$prefix.res_blocks.$layer"
                _ltx_vae_conv_specs!(specs, "$base.conv1.conv", channels, channels)
                _ltx_vae_conv_specs!(specs, "$base.conv2.conv", channels, channels)
            end
        else
            stride = block.kind === :compress_space_res ? (1, 2, 2) :
                block.kind === :compress_time_res ? (2, 1, 1) :
                block.kind === :compress_all_res ? (2, 2, 2) :
                throw(ArgumentError("unsupported LTX VAE block $(block.kind)"))
            output = channels * block.multiplier
            convolution_output = output ÷ prod(stride)
            output % prod(stride) == 0 ||
                throw(ArgumentError("LTX VAE downsample channels are not divisible by stride"))
            _ltx_vae_conv_specs!(specs, "$prefix.conv.conv",
                convolution_output, channels)
            channels = output
        end
    end
    output_channels = config.latent_log_variance === :per_channel ?
        2config.latent_channels :
        config.latent_log_variance in (:uniform, :constant) ?
            config.latent_channels + 1 : config.latent_channels
    _ltx_vae_conv_specs!(specs, checkpoint_prefix * "conv_out.conv",
        output_channels, channels)
    push!(specs,
        TensorSpec(statistics_prefix * "mean-of-means",
                   [config.latent_channels], VECTOR_LAYOUT),
        TensorSpec(statistics_prefix * "std-of-means",
                   [config.latent_channels], VECTOR_LAYOUT))
    specs
end

abstract type AbstractLTXVAEEncoderBlock end

struct LTXVAEResidualBlock{A,B} <: AbstractLTXVAEEncoderBlock
    conv1::A
    conv2::B
end

struct LTXVAEDownsample{C} <: AbstractLTXVAEEncoderBlock
    conv::C
    stride::NTuple{3,Int}
    output_channels::Int
end

struct LTXVideoVAEEncoder{I,O,M,S}
    config::LTXVideoVAEConfig
    conv_in::I
    blocks::Vector{AbstractLTXVAEEncoderBlock}
    conv_out::O
    mean_of_means::M
    std_of_means::S
end

function _new_ltx_vae_conv(rng, output, input; initialize=true)
    weight = initialize ?
        randn(rng, Float32, output, input, 3, 3, 3) ./
            sqrt(Float32(input * 27)) :
        zeros(Float32, output, input, 3, 3, 3)
    VAEConv3D(weight, zeros(Float32, output); padding=(1, 1, 1))
end

function LTXVideoVAEEncoder(config::LTXVideoVAEConfig=LTXVideoVAEConfig();
                            rng=Random.default_rng(), initialize=true)
    channels = config.latent_channels
    conv_in = _new_ltx_vae_conv(rng, channels,
        config.input_channels * config.patch_size^2; initialize=initialize)
    blocks = AbstractLTXVAEEncoderBlock[]
    for block in config.blocks
        if block.kind === :res_x
            for _ in 1:block.num_layers
                push!(blocks, LTXVAEResidualBlock(
                    _new_ltx_vae_conv(rng, channels, channels;
                                      initialize=initialize),
                    _new_ltx_vae_conv(rng, channels, channels;
                                      initialize=initialize)))
            end
        else
            stride = block.kind === :compress_space_res ? (1, 2, 2) :
                block.kind === :compress_time_res ? (2, 1, 1) :
                block.kind === :compress_all_res ? (2, 2, 2) :
                throw(ArgumentError("unsupported LTX VAE block $(block.kind)"))
            output = channels * block.multiplier
            output % prod(stride) == 0 ||
                throw(ArgumentError("LTX VAE downsample channels are not divisible by stride"))
            push!(blocks, LTXVAEDownsample(
                _new_ltx_vae_conv(rng, output ÷ prod(stride), channels;
                                  initialize=initialize),
                stride, output))
            channels = output
        end
    end
    output_channels = config.latent_log_variance === :per_channel ?
        2config.latent_channels :
        config.latent_log_variance in (:uniform, :constant) ?
            config.latent_channels + 1 : config.latent_channels
    LTXVideoVAEEncoder(config, conv_in, blocks,
        _new_ltx_vae_conv(rng, output_channels, channels;
                          initialize=initialize),
        zeros(Float32, config.latent_channels),
        ones(Float32, config.latent_channels))
end

function ltx23_vae_patchify(video::AbstractArray{T,5},
                            patch_size::Integer) where T
    channels, frames, height, width, batch = size(video)
    patch_size > 0 || throw(ArgumentError("LTX VAE patch size must be positive"))
    height % patch_size == 0 && width % patch_size == 0 ||
        throw(DimensionMismatch("LTX video dimensions must divide VAE patch size"))
    split = reshape(video, channels, frames, patch_size,
        height ÷ patch_size, patch_size, width ÷ patch_size, batch)
    ordered = permutedims(split, (3, 5, 1, 2, 4, 6, 7))
    reshape(ordered, channels * patch_size^2, frames,
        height ÷ patch_size, width ÷ patch_size, batch)
end

function ltx23_pixel_norm(x::AbstractArray; epsilon=1f-8)
    x32 = float32_values(x)
    normalized = x32 ./ sqrt.(sum(abs2, x32; dims=1) ./
                              Float32(size(x, 1)) .+ Float32(epsilon))
    cast_values(eltype(x), normalized)
end

function ltx23_causal_conv3d(layer::VAEConv3D, x::AbstractArray)
    temporal_padding = size(layer.weight, 3) - 1
    padded = temporal_padding == 0 ? x :
        cat(repeat(view(x, :, 1:1, :, :, :);
                   inner=(1, temporal_padding, 1, 1, 1)), x; dims=2)
    vae_conv3d(layer, padded; temporal_left=0, temporal_right=0)
end

function ltx23_vae_residual_forward(block::LTXVAEResidualBlock,
                                    x::AbstractArray; epsilon=1f-8)
    hidden = ltx23_causal_conv3d(block.conv1,
        vae_silu(ltx23_pixel_norm(x; epsilon=epsilon)))
    hidden = ltx23_causal_conv3d(block.conv2,
        vae_silu(ltx23_pixel_norm(hidden; epsilon=epsilon)))
    hidden .+ x
end

function _ltx23_space_to_depth(x::AbstractArray,
                               stride::NTuple{3,Int})
    channels, frames, height, width, batch = size(x)
    pt, ph, pw = stride
    frames % pt == 0 && height % ph == 0 && width % pw == 0 ||
        throw(DimensionMismatch("LTX VAE downsample input does not divide stride"))
    split = reshape(x, channels, pt, frames ÷ pt, ph, height ÷ ph,
                    pw, width ÷ pw, batch)
    ordered = permutedims(split, (6, 4, 2, 1, 7, 5, 3, 8))
    reshape(ordered, channels * prod(stride), frames ÷ pt,
        height ÷ ph, width ÷ pw, batch)
end

function ltx23_vae_downsample_forward(block::LTXVAEDownsample,
                                      x::AbstractArray)
    input = block.stride[1] == 2 ?
        cat(view(x, :, 1:1, :, :, :), x; dims=2) : x
    shortcut = _ltx23_space_to_depth(input, block.stride)
    group_size = size(shortcut, 1) ÷ block.output_channels
    shortcut = dropdims(sum(reshape(shortcut, group_size,
        block.output_channels, size(shortcut)[2:end]...); dims=1);
        dims=1) ./ eltype(shortcut)(group_size)
    convolved = _ltx23_space_to_depth(
        ltx23_causal_conv3d(block.conv, input), block.stride)
    convolved .+ shortcut
end

function ltx23_vae_encoder_forward(model::LTXVideoVAEEncoder,
                                   video::AbstractArray{T,5}) where T
    size(video, 1) == model.config.input_channels ||
        throw(DimensionMismatch("LTX VAE input channel count differs"))
    frames_to_crop = (size(video, 2) - 1) % LTX23_VIDEO_SCALE_FACTORS[1]
    input = frames_to_crop == 0 ? video :
        view(video, :, 1:size(video, 2)-frames_to_crop, :, :, :)
    hidden = ltx23_causal_conv3d(model.conv_in,
        ltx23_vae_patchify(input, model.config.patch_size))
    for block in model.blocks
        hidden = block isa LTXVAEResidualBlock ?
            ltx23_vae_residual_forward(block, hidden;
                epsilon=model.config.epsilon) :
            ltx23_vae_downsample_forward(block, hidden)
    end
    output = ltx23_causal_conv3d(model.conv_out,
        vae_silu(ltx23_pixel_norm(hidden; epsilon=model.config.epsilon)))
    means = if model.config.latent_log_variance === :per_channel
        view(output, 1:model.config.latent_channels, :, :, :, :)
    elseif model.config.latent_log_variance in (:uniform, :constant)
        view(output, 1:size(output, 1)-1, :, :, :, :)
    else
        output
    end
    shape = (:, 1, 1, 1, 1)
    (means .- reshape(cast_values(eltype(means), model.mean_of_means), shape...)) ./
        reshape(cast_values(eltype(means), model.std_of_means), shape...)
end

function _load_ltx_vae_conv!(source, prefix, layer)
    _load_vae_conv!(source, prefix, layer)
end

function load_ltx23_vae_encoder(source::AbstractTensorSource,
                                config::LTXVideoVAEConfig=LTXVideoVAEConfig();
                                strict=false)
    audit = audit_state_dict(source, ltx23_vae_encoder_specs(config);
        allow_unexpected=!strict)
    isempty(audit.missing) && isempty(audit.shape_mismatches) ||
        _assert_clean_audit(StateDictAudit(audit.missing, String[],
                                          audit.shape_mismatches))
    model = LTXVideoVAEEncoder(config; rng=Xoshiro(0), initialize=false)
    _load_ltx_vae_conv!(source, "vae.encoder.conv_in.conv", model.conv_in)
    flattened = 1
    for (block_index, block_config) in enumerate(config.blocks)
        prefix = "vae.encoder.down_blocks.$(block_index - 1)"
        if block_config.kind === :res_x
            for layer in 0:block_config.num_layers-1
                block = model.blocks[flattened]::LTXVAEResidualBlock
                _load_ltx_vae_conv!(source,
                    "$prefix.res_blocks.$layer.conv1.conv", block.conv1)
                _load_ltx_vae_conv!(source,
                    "$prefix.res_blocks.$layer.conv2.conv", block.conv2)
                flattened += 1
            end
        else
            block = model.blocks[flattened]::LTXVAEDownsample
            _load_ltx_vae_conv!(source, "$prefix.conv.conv", block.conv)
            flattened += 1
        end
    end
    _load_ltx_vae_conv!(source, "vae.encoder.conv_out.conv", model.conv_out)
    copyto!(model.mean_of_means, load_state_tensor(source,
        TensorSpec("vae.per_channel_statistics.mean-of-means",
                   [config.latent_channels], VECTOR_LAYOUT)))
    copyto!(model.std_of_means, load_state_tensor(source,
        TensorSpec("vae.per_channel_statistics.std-of-means",
                   [config.latent_channels], VECTOR_LAYOUT)))
    model
end

load_ltx23_vae_encoder(path::AbstractString;
                       config=ltx23_vae_config(path), kwargs...) =
    load_ltx23_vae_encoder(open_tensor_source(path), config; kwargs...)

function move_to_device(model::LTXVideoVAEEncoder, transfer)
    blocks = AbstractLTXVAEEncoderBlock[
        block isa LTXVAEResidualBlock ?
            LTXVAEResidualBlock(_move_vae_conv(block.conv1, transfer),
                                _move_vae_conv(block.conv2, transfer)) :
            LTXVAEDownsample(_move_vae_conv(block.conv, transfer),
                             block.stride, block.output_channels)
        for block in model.blocks
    ]
    LTXVideoVAEEncoder(model.config, _move_vae_conv(model.conv_in, transfer),
        blocks, _move_vae_conv(model.conv_out, transfer),
        transfer(model.mean_of_means), transfer(model.std_of_means))
end

move_to_device(model::LTXVideoVAEEncoder, device::Symbol, precision::Symbol) =
    move_to_device(model, array_transfer(device, precision))
