struct WanAttention{Q,K,V,O,NQ,NK,KI,VI,NKI}
    q::Q
    k::K
    v::V
    o::O
    norm_q::NQ
    norm_k::NK
    k_img::KI
    v_img::VI
    norm_k_img::NKI
    heads::Int
end

function WanAttention(dim::Integer, heads::Integer; rng=Random.default_rng(),
                      epsilon=1f-6, initialize=true, image_context=false)
    dim % heads == 0 || throw(ArgumentError("attention dimension must divide heads"))
    init() = begin
        weight = initialize ?
            randn(rng, Float32, dim, dim) ./ sqrt(Float32(dim)) :
            zeros(Float32, dim, dim)
        DenseLayer(weight, zeros(Float32, dim))
    end
    WanAttention(init(), init(), init(), init(), RMSNorm(dim; epsilon=epsilon),
        RMSNorm(dim; epsilon=epsilon),
        image_context ? init() : nothing,
        image_context ? init() : nothing,
        image_context ? RMSNorm(dim; epsilon=epsilon) : nothing,
        heads)
end

struct WanTransformerBlock{SA,CA,F1,F2,M,NW,NB}
    self_attention::SA
    cross_attention::CA
    ffn_in::F1
    ffn_out::F2
    modulation::M       # (features, 6)
    cross_norm_weight::NW
    cross_norm_bias::NB
    epsilon::Float32
end

function WanTransformerBlock(dim::Integer, ffn_dim::Integer, heads::Integer;
                             rng=Random.default_rng(), epsilon=1f-6,
                             cross_attention_norm=true, initialize=true,
                             image_context=false)
    init(out, input) = DenseLayer(
        initialize ? randn(rng, Float32, out, input) ./ sqrt(Float32(input)) :
            zeros(Float32, out, input),
        zeros(Float32, out))
    WanTransformerBlock(WanAttention(dim, heads; rng=rng, epsilon=epsilon,
                                     initialize=initialize),
        WanAttention(dim, heads; rng=rng, epsilon=epsilon,
                     initialize=initialize, image_context=image_context),
        init(ffn_dim, dim), init(dim, ffn_dim),
        initialize ? randn(rng, Float32, dim, 6) ./ sqrt(Float32(dim)) :
            zeros(Float32, dim, 6),
        cross_attention_norm ? ones(Float32, dim) : nothing,
        cross_attention_norm ? zeros(Float32, dim) : nothing,
        Float32(epsilon))
end

struct WanImageProjection{NW,NB,D1,D2,OW,OB}
    input_norm_weight::NW
    input_norm_bias::NB
    input_projection::D1
    output_projection::D2
    output_norm_weight::OW
    output_norm_bias::OB
    epsilon::Float32
end

function WanImageProjection(dim::Integer; rng=Random.default_rng(),
                            initialize=true, epsilon=1f-5)
    dense(out, input) = DenseLayer(
        initialize ?
            randn(rng, Float32, out, input) ./ sqrt(Float32(input)) :
            zeros(Float32, out, input),
        zeros(Float32, out))
    WanImageProjection(
        ones(Float32, 1280), zeros(Float32, 1280),
        dense(1280, 1280), dense(dim, 1280),
        ones(Float32, dim), zeros(Float32, dim), Float32(epsilon))
end

function wan_image_projection_forward(projection::WanImageProjection,
                                      features::AbstractArray{T,3};
                                      training::Bool=false,
                                      dropout_seed::UInt64=UInt64(0)) where T
    size(features, 1) == 1280 ||
        throw(DimensionMismatch("Wan image features must have 1280 channels"))
    normalized = layernorm(features; epsilon=projection.epsilon,
        weight=projection.input_norm_weight,
        bias=projection.input_norm_bias)
    hidden = _projection(projection.input_projection, normalized;
        training=training, dropout_seed=dropout_seed, dropout_stream=1)
    hidden32 = float32_values(hidden)
    activated32 = NNlib.gelu.(hidden32)
    activated = cast_values(eltype(hidden), activated32)
    output = _projection(projection.output_projection, activated;
        training=training, dropout_seed=dropout_seed, dropout_stream=2)
    layernorm(output; epsilon=projection.epsilon,
        weight=projection.output_norm_weight,
        bias=projection.output_norm_bias)
end

function _split_heads(x, heads)
    dim, tokens, batch = size(x)
    dim % heads == 0 || throw(DimensionMismatch("feature count does not divide heads"))
    reshape(x, dim ÷ heads, heads, tokens, batch)
end
_join_heads(x) = reshape(x, size(x, 1) * size(x, 2), size(x, 3), size(x, 4))

"""
Wan's factorized 3D RoPE. `grid_sizes[batch]` is `(frames, height, width)`;
tokens are ordered with width varying fastest, matching official patch flattening.
"""
function wan_rope3d(x::AbstractArray{T,4},
                    grid_sizes::AbstractVector{<:NTuple{3,Int}};
                    base=10_000f0, inverse=false) where T
    length(grid_sizes) == size(x, 4) ||
        throw(DimensionMismatch("grid count and batch count differ"))
    iseven(size(x, 1)) || throw(DimensionMismatch("Wan RoPE head dimension must be even"))
    total_pairs = size(x, 1) ÷ 2
    spatial_pairs = total_pairs ÷ 3
    pair_counts = (total_pairs - 2spatial_pairs, spatial_pairs, spatial_pairs)
    direction = inverse ? -1f0 : 1f0
    # Build the small position/frequency table on the host, then perform the
    # elementwise rotation on the input device. This avoids GPU scalar indexing
    # while keeping variable per-sample grids and padded tokens supported.
    angles_host = zeros(Float32, total_pairs, size(x, 3), size(x, 4))
    for batch in axes(x, 4)
        frames, height, width = grid_sizes[batch]
        sequence = frames * height * width
        sequence <= size(x, 3) ||
            throw(DimensionMismatch("grid has more tokens than the input"))
        for token0 in 0:sequence-1
            frame = token0 ÷ (height * width)
            remainder = token0 % (height * width)
            row, column = remainder ÷ width, remainder % width
            coordinates = (frame, row, column)
            pair_offset = 0
            for axis in 1:3
                count = pair_counts[axis]
                for local_pair in 0:count-1
                    pair = pair_offset + local_pair
                    theta = direction * Float32(coordinates[axis]) /
                        Float32(base)^Float32(local_pair / count)
                    angles_host[pair + 1, token0 + 1, batch] = theta
                end
                pair_offset += count
            end
        end
    end
    angles = similar(x, Float32, size(angles_host))
    copyto!(angles, angles_host)
    c = reshape(cos.(angles), total_pairs, 1, size(x, 3), size(x, 4))
    s = reshape(sin.(angles), total_pairs, 1, size(x, 3), size(x, 4))
    odd = @view x[1:2:size(x, 1), :, :, :]
    even = @view x[2:2:size(x, 1), :, :, :]
    odd32 = float32_values(odd)
    even32 = float32_values(even)
    rotated_odd = c .* odd32 .- s .* even32
    rotated_even = s .* odd32 .+ c .* even32
    y = similar(x)
    destination_odd = @view y[1:2:size(x, 1), :, :, :]
    destination_even = @view y[2:2:size(x, 1), :, :, :]
    if T === BFloat16
        copyto!(destination_odd, bfloat16_values(rotated_odd))
        copyto!(destination_even, bfloat16_values(rotated_even))
    else
        copyto!(destination_odd, cast_values(T, rotated_odd))
        copyto!(destination_even, cast_values(T, rotated_even))
    end
    y
end

function _wan_attention(attention::WanAttention, x, context=x;
                        grid_sizes=nothing, rope=false,
                        training::Bool=false,
                        dropout_seed::UInt64=UInt64(0),
                        dropout_stream::Integer=0)
    q_flat = attention.norm_q(_projection(attention.q, x;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 1))
    k_flat = attention.norm_k(_projection(attention.k, context;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 2))
    v_flat = _projection(attention.v, context;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 3)
    q, k, v = _split_heads(q_flat, attention.heads),
              _split_heads(k_flat, attention.heads),
              _split_heads(v_flat, attention.heads)
    if rope
        grid_sizes === nothing && throw(ArgumentError("self attention requires grid sizes"))
        q = wan_rope3d(q, grid_sizes)
        k = wan_rope3d(k, grid_sizes)
    end
    attended = memory_efficient_attention(q, k, v).output
    _projection(attention.o, _join_heads(attended);
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 4)
end

function _wan_i2v_cross_attention(attention::WanAttention, x, context,
                                  text_length::Integer;
                                  training::Bool=false,
                                  dropout_seed::UInt64=UInt64(0),
                                  dropout_stream::Integer=0)
    attention.k_img === nothing &&
        throw(ArgumentError("Wan I2V attention lacks image projections"))
    size(context, 2) > text_length ||
        throw(DimensionMismatch(
            "Wan I2V context must prepend at least one image token"))
    image_tokens = size(context, 2) - Int(text_length)
    # Materialized slices preserve a device-native dense layout. A reshaped
    # SubArray falls back to scalar host iteration in CUDA's LoRA matmuls.
    image_context = context[:, 1:image_tokens, :]
    text_context = context[:, image_tokens + 1:end, :]

    q = _split_heads(attention.norm_q(_projection(attention.q, x;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 1)), attention.heads)
    text_k = _split_heads(
        attention.norm_k(_projection(attention.k, text_context;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=dropout_stream + 2)), attention.heads)
    text_v = _split_heads(_projection(attention.v, text_context;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 3), attention.heads)
    image_k = _split_heads(
        attention.norm_k_img(_projection(attention.k_img, image_context;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=dropout_stream + 4)), attention.heads)
    image_v = _split_heads(_projection(attention.v_img, image_context;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 5), attention.heads)
    text_output = memory_efficient_attention(q, text_k, text_v).output
    image_output = memory_efficient_attention(q, image_k, image_v).output
    _projection(attention.o, mixed_add(
        _join_heads(text_output), _join_heads(image_output));
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 6)
end

"""
Forward through one Wan 2.1 transformer block in canonical layout:
`x=(features,tokens,batch)`, `time_modulation=(features,6,batch)`, and
`context=(features,text_tokens,batch)`.
"""
function wan_block_forward(block::WanTransformerBlock, x::AbstractArray{T,3},
                           time_modulation::AbstractArray{S,3},
                           context::AbstractArray{U,3},
                           grid_sizes::AbstractVector{<:NTuple{3,Int}},
                           image_context_tokens::Int=0,
                           training::Bool=false,
                           dropout_seed::UInt64=UInt64(0)) where {T,S,U}
    dim, _, batch = size(x)
    size(time_modulation) == (dim, 6, batch) ||
        throw(DimensionMismatch("time modulation must be (features,6,batch)"))
    size(context, 1) == dim && size(context, 3) == batch ||
        throw(DimensionMismatch("context dimensions differ"))
    e = float32_values(time_modulation) .+
        reshape(float32_values(block.modulation), dim, 6, 1)
    part(i) = reshape(e[:, i, :], dim, 1, batch)

    normalized = layernorm(x; epsilon=block.epsilon)
    self_input = mixed_affine(normalized, part(2), part(1))
    self_output = _wan_attention(block.self_attention, self_input;
        grid_sizes=grid_sizes, rope=true, training=training,
        dropout_seed=dropout_seed, dropout_stream=10)
    x = mixed_add(x, mixed_mul(self_output, part(3)))

    cross_input = layernorm(x; epsilon=block.epsilon,
        weight=block.cross_norm_weight, bias=block.cross_norm_bias)
    cross_output = block.cross_attention.k_img === nothing ?
        _wan_attention(block.cross_attention, cross_input, context;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=20) :
        _wan_i2v_cross_attention(
            block.cross_attention, cross_input, context,
            size(context, 2) - image_context_tokens;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=20)
    x = mixed_add(x, cross_output)

    ffn_input = mixed_affine(
        layernorm(x; epsilon=block.epsilon), part(5), part(4))
    hidden = _projection(block.ffn_in, ffn_input;
        training=training, dropout_seed=dropout_seed, dropout_stream=30)
    ffn_output = _projection(block.ffn_out, gelu_tanh(hidden);
        training=training, dropout_seed=dropout_seed, dropout_stream=31)
    mixed_add(x, mixed_mul(ffn_output, part(6)))
end

function _load_dense!(source, prefix, layer::DenseLayer)
    _copy_model_tensor!(layer.weight, load_state_tensor(source,
        TensorSpec("$prefix.weight", collect(size(layer.weight)), LINEAR_OUT_IN)))
    layer.bias === nothing || _copy_model_tensor!(layer.bias, load_state_tensor(source,
        TensorSpec("$prefix.bias", [length(layer.bias)], VECTOR_LAYOUT)))
end

function _load_attention!(source, prefix, attention::WanAttention)
    _load_dense!(source, "$prefix.q", attention.q)
    _load_dense!(source, "$prefix.k", attention.k)
    _load_dense!(source, "$prefix.v", attention.v)
    _load_dense!(source, "$prefix.o", attention.o)
    _copy_model_tensor!(attention.norm_q.weight, load_state_tensor(source,
        TensorSpec("$prefix.norm_q.weight", [length(attention.norm_q.weight)],
                   VECTOR_LAYOUT)))
    _copy_model_tensor!(attention.norm_k.weight, load_state_tensor(source,
        TensorSpec("$prefix.norm_k.weight", [length(attention.norm_k.weight)],
                   VECTOR_LAYOUT)))
    if attention.k_img !== nothing
        _load_dense!(source, "$prefix.k_img", attention.k_img)
        _load_dense!(source, "$prefix.v_img", attention.v_img)
        _copy_model_tensor!(attention.norm_k_img.weight, load_state_tensor(source,
            TensorSpec("$prefix.norm_k_img.weight",
                [length(attention.norm_k_img.weight)], VECTOR_LAYOUT)))
    end
end

function _load_block!(source, root, block::WanTransformerBlock,
                      config::Wan21Config)
    _load_attention!(source, "$root.self_attn", block.self_attention)
    _load_attention!(source, "$root.cross_attn", block.cross_attention)
    _load_dense!(source, "$root.ffn.0", block.ffn_in)
    _load_dense!(source, "$root.ffn.2", block.ffn_out)
    official_modulation = load_state_tensor(source,
        TensorSpec("$root.modulation", [1, 6, config.hidden_size],
                   ROW_MAJOR_SOURCE))
    _copy_model_tensor!(block.modulation,
        permutedims(dropdims(official_modulation; dims=1)))
    if config.cross_attention_norm
        _copy_model_tensor!(block.cross_norm_weight, load_state_tensor(source,
            TensorSpec("$root.norm3.weight", [config.hidden_size], VECTOR_LAYOUT)))
        _copy_model_tensor!(block.cross_norm_bias, load_state_tensor(source,
            TensorSpec("$root.norm3.bias", [config.hidden_size], VECTOR_LAYOUT)))
    end
    block
end

function _load_image_projection!(source,
                                 projection::WanImageProjection,
                                 config::Wan21Config)
    _copy_model_tensor!(projection.input_norm_weight, load_state_tensor(source,
        TensorSpec("img_emb.proj.0.weight", [1280], VECTOR_LAYOUT)))
    _copy_model_tensor!(projection.input_norm_bias, load_state_tensor(source,
        TensorSpec("img_emb.proj.0.bias", [1280], VECTOR_LAYOUT)))
    _load_dense!(source, "img_emb.proj.1", projection.input_projection)
    _load_dense!(source, "img_emb.proj.3", projection.output_projection)
    _copy_model_tensor!(projection.output_norm_weight, load_state_tensor(source,
        TensorSpec("img_emb.proj.4.weight",
            [config.hidden_size], VECTOR_LAYOUT)))
    _copy_model_tensor!(projection.output_norm_bias, load_state_tensor(source,
        TensorSpec("img_emb.proj.4.bias",
            [config.hidden_size], VECTOR_LAYOUT)))
    projection
end

"""Load one Wan transformer block from official-format SafeTensors weights."""
function load_wan_block(path::AbstractString, prefix::AbstractString,
                        config::Wan21Config; block_index=0)
    source = open_tensor_source(path)
    block = WanTransformerBlock(config.hidden_size, config.ffn_size, config.heads;
        epsilon=config.epsilon,
        cross_attention_norm=config.cross_attention_norm,
        rng=Xoshiro(0), initialize=false)
    root = "$prefix$block_index"
    _load_block!(source, root, block, config)
end

struct WanTransformer{PW,PB,TE1,TE2,TI1,TI2,TP,BS,H,HM,IP}
    config::Wan21Config
    patch_weight::PW
    patch_bias::PB
    text_in::TE1
    text_out::TE2
    time_in::TI1
    time_out::TI2
    time_projection::TP
    blocks::BS
    head::H
    head_modulation::HM
    image_projection::IP
end

_move_array(x, transfer) = x === nothing ? nothing : transfer(x)
_move_dense(layer::DenseLayer, transfer) =
    DenseLayer(_move_weight(layer.weight, transfer),
               _move_array(layer.bias, transfer))
_move_norm(norm::RMSNorm, transfer) = RMSNorm(transfer(norm.weight), norm.epsilon)
function _move_attention(attention::WanAttention, transfer)
    WanAttention(_move_dense(attention.q, transfer),
        _move_dense(attention.k, transfer), _move_dense(attention.v, transfer),
        _move_dense(attention.o, transfer), _move_norm(attention.norm_q, transfer),
        _move_norm(attention.norm_k, transfer),
        attention.k_img === nothing ? nothing :
            _move_dense(attention.k_img, transfer),
        attention.v_img === nothing ? nothing :
            _move_dense(attention.v_img, transfer),
        attention.norm_k_img === nothing ? nothing :
            _move_norm(attention.norm_k_img, transfer),
        attention.heads)
end
function _move_block(block::WanTransformerBlock, transfer)
    WanTransformerBlock(_move_attention(block.self_attention, transfer),
        _move_attention(block.cross_attention, transfer),
        _move_dense(block.ffn_in, transfer), _move_dense(block.ffn_out, transfer),
        transfer(block.modulation), _move_array(block.cross_norm_weight, transfer),
        _move_array(block.cross_norm_bias, transfer), block.epsilon)
end

function _move_image_projection(projection::WanImageProjection, transfer)
    WanImageProjection(
        transfer(projection.input_norm_weight),
        transfer(projection.input_norm_bias),
        _move_dense(projection.input_projection, transfer),
        _move_dense(projection.output_projection, transfer),
        transfer(projection.output_norm_weight),
        transfer(projection.output_norm_bias),
        projection.epsilon)
end

"""
    move_to_device(model, device)

Copy every Wan transformer parameter to `:cpu` or `:cuda`. A callable array
transfer may also be supplied for custom devices or storage types.
"""
function move_to_device(model::WanTransformer, transfer)
    WanTransformer(model.config, transfer(model.patch_weight),
        transfer(model.patch_bias), _move_dense(model.text_in, transfer),
        _move_dense(model.text_out, transfer), _move_dense(model.time_in, transfer),
        _move_dense(model.time_out, transfer),
        _move_dense(model.time_projection, transfer),
        [_move_block(block, transfer) for block in model.blocks],
        _move_dense(model.head, transfer), transfer(model.head_modulation),
        model.image_projection === nothing ? nothing :
            _move_image_projection(model.image_projection, transfer))
end
move_to_device(model::WanTransformer, ::Val{:cpu}) =
    move_to_device(model, Array)
function move_to_device(model::WanTransformer, ::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    move_to_device(model, CUDA.CuArray)
end
move_to_device(model::WanTransformer, device::Symbol) =
    device === :cpu ? move_to_device(model, Val(:cpu)) :
    device === :cuda ? move_to_device(model, Val(:cuda)) :
    throw(ArgumentError("unsupported device: $device"))

function _move_wan_bf16(model::WanTransformer, device::Symbol)
    regular = array_transfer(device, :bf16)
    fp32_transfer = array_transfer(device, :fp32)
    # Match upstream autocast: parameters are first rounded to BF16 storage,
    # then the time/modulation math consumes those stored values in FP32.
    control(array) = fp32_transfer(
        float32_values(bfloat16_values(Array(array))))
    move_block(block) = WanTransformerBlock(
        _move_attention(block.self_attention, regular),
        _move_attention(block.cross_attention, regular),
        _move_dense(block.ffn_in, regular), _move_dense(block.ffn_out, regular),
        control(block.modulation),
        _move_array(block.cross_norm_weight, regular),
        _move_array(block.cross_norm_bias, regular), block.epsilon)
    # Upstream stores the modulation tables in BF16 but promotes their values
    # to FP32 before combining them with the time-conditioning gates.
    WanTransformer(model.config, regular(model.patch_weight),
        regular(model.patch_bias), _move_dense(model.text_in, regular),
        _move_dense(model.text_out, regular), _move_dense(model.time_in, regular),
        _move_dense(model.time_out, regular),
        _move_dense(model.time_projection, regular),
        [move_block(block) for block in model.blocks],
        _move_dense(model.head, regular), control(model.head_modulation),
        model.image_projection === nothing ? nothing :
            _move_image_projection(model.image_projection, regular))
end

function move_to_device(model::WanTransformer, device::Symbol,
                        precision::Symbol)
    precision === :bf16 && return _move_wan_bf16(model, device)
    move_to_device(model, array_transfer(device, precision))
end

function WanTransformer(config::Wan21Config; rng=Random.default_rng(), initialize=true)
    d = config.hidden_size
    dense(out, input; zero=false) = DenseLayer(
        (zero || !initialize) ? zeros(Float32, out, input) :
            randn(rng, Float32, out, input) ./ sqrt(Float32(input)),
        zeros(Float32, out))
    patch_weight = initialize ?
        randn(rng, Float32, d, config.input_channels, config.patch_size...) ./
            sqrt(Float32(config.input_channels * prod(config.patch_size))) :
        zeros(Float32, d, config.input_channels, config.patch_size...)
    blocks = [WanTransformerBlock(d, config.ffn_size, config.heads;
        rng=rng, epsilon=config.epsilon,
        cross_attention_norm=config.cross_attention_norm, initialize=initialize,
        image_context=config.model_type === :i2v)
        for _ in 1:config.layers]
    head_output = prod(config.patch_size) * config.output_channels
    WanTransformer(config, patch_weight, zeros(Float32, d),
        dense(d, config.text_size), dense(d, d),
        dense(d, config.frequency_size), dense(d, d), dense(6d, d),
        blocks, dense(head_output, d; zero=true),
        initialize ? randn(rng, Float32, d, 2) ./ sqrt(Float32(d)) :
            zeros(Float32, d, 2),
        config.model_type === :i2v ?
            WanImageProjection(
                d; rng=rng, initialize=initialize) : nothing)
end

function _assert_clean_audit(audit::StateDictAudit)
    isempty(audit) && return
    messages = String[]
    isempty(audit.missing) ||
        push!(messages, "missing tensors: " * join(audit.missing, ", "))
    isempty(audit.unexpected) ||
        push!(messages, "unexpected tensors: " * join(audit.unexpected, ", "))
    isempty(audit.shape_mismatches) ||
        push!(messages, "shape mismatches: " * join(audit.shape_mismatches, "; "))
    throw(ArgumentError(join(messages, '\n')))
end

"""
Stream an official-format Wan transformer state dict into native Julia arrays.
Only one source tensor is resident in temporary memory at a time.
"""
function load_wan_transformer(source::AbstractTensorSource, config::Wan21Config;
                              strict=true)
    specs = wan21_transformer_specs(config)
    audit = audit_state_dict(source, specs; allow_unexpected=!strict)
    _assert_clean_audit(audit)
    model = WanTransformer(config; rng=Xoshiro(0), initialize=false)
    _copy_model_tensor!(model.patch_weight, load_state_tensor(source,
        TensorSpec("patch_embedding.weight",
            [config.hidden_size, config.input_channels, config.patch_size...],
            CONV_OUT_IN_SPATIAL)))
    _copy_model_tensor!(model.patch_bias, load_state_tensor(source,
        TensorSpec("patch_embedding.bias", [config.hidden_size], VECTOR_LAYOUT)))
    _load_dense!(source, "text_embedding.0", model.text_in)
    _load_dense!(source, "text_embedding.2", model.text_out)
    _load_dense!(source, "time_embedding.0", model.time_in)
    _load_dense!(source, "time_embedding.2", model.time_out)
    _load_dense!(source, "time_projection.1", model.time_projection)
    for (index, block) in enumerate(model.blocks)
        _load_block!(source, "blocks.$(index - 1)", block, config)
    end
    model.image_projection === nothing ||
        _load_image_projection!(source, model.image_projection, config)
    official_head_modulation = load_state_tensor(source,
        TensorSpec("head.modulation", [1, 2, config.hidden_size],
                   ROW_MAJOR_SOURCE))
    _copy_model_tensor!(model.head_modulation,
        permutedims(dropdims(official_head_modulation; dims=1)))
    _load_dense!(source, "head.head", model.head)
    model
end

load_wan_transformer(path::AbstractString, config::Wan21Config; kwargs...) =
    load_wan_transformer(open_tensor_source(path), config; kwargs...)

function _store_dense!(state, prefix, layer::DenseLayer)
    state["$prefix.weight"] = copy(layer.weight)
    layer.bias === nothing || (state["$prefix.bias"] = copy(layer.bias))
end
function _store_attention!(state, prefix, attention::WanAttention)
    _store_dense!(state, "$prefix.q", attention.q)
    _store_dense!(state, "$prefix.k", attention.k)
    _store_dense!(state, "$prefix.v", attention.v)
    _store_dense!(state, "$prefix.o", attention.o)
    state["$prefix.norm_q.weight"] = copy(attention.norm_q.weight)
    state["$prefix.norm_k.weight"] = copy(attention.norm_k.weight)
    if attention.k_img !== nothing
        _store_dense!(state, "$prefix.k_img", attention.k_img)
        _store_dense!(state, "$prefix.v_img", attention.v_img)
        state["$prefix.norm_k_img.weight"] =
            copy(attention.norm_k_img.weight)
    end
end

"""Return a standards-layout official-key state dict for a native Wan model."""
function wan_transformer_state_dict(model::WanTransformer)
    state = Dict{String,AbstractArray}()
    state["patch_embedding.weight"] = copy(model.patch_weight)
    state["patch_embedding.bias"] = copy(model.patch_bias)
    _store_dense!(state, "text_embedding.0", model.text_in)
    _store_dense!(state, "text_embedding.2", model.text_out)
    _store_dense!(state, "time_embedding.0", model.time_in)
    _store_dense!(state, "time_embedding.2", model.time_out)
    _store_dense!(state, "time_projection.1", model.time_projection)
    for (index, block) in enumerate(model.blocks)
        root = "blocks.$(index - 1)"
        _store_attention!(state, "$root.self_attn", block.self_attention)
        _store_attention!(state, "$root.cross_attn", block.cross_attention)
        _store_dense!(state, "$root.ffn.0", block.ffn_in)
        _store_dense!(state, "$root.ffn.2", block.ffn_out)
        state["$root.modulation"] =
            reshape(permutedims(block.modulation), 1, 6, model.config.hidden_size)
        if model.config.cross_attention_norm
            state["$root.norm3.weight"] = copy(block.cross_norm_weight)
            state["$root.norm3.bias"] = copy(block.cross_norm_bias)
        end
    end
    if model.image_projection !== nothing
        projection = model.image_projection
        state["img_emb.proj.0.weight"] =
            copy(projection.input_norm_weight)
        state["img_emb.proj.0.bias"] =
            copy(projection.input_norm_bias)
        _store_dense!(
            state, "img_emb.proj.1", projection.input_projection)
        _store_dense!(
            state, "img_emb.proj.3", projection.output_projection)
        state["img_emb.proj.4.weight"] =
            copy(projection.output_norm_weight)
        state["img_emb.proj.4.bias"] =
            copy(projection.output_norm_bias)
    end
    state["head.modulation"] =
        reshape(permutedims(model.head_modulation), 1, 2, model.config.hidden_size)
    _store_dense!(state, "head.head", model.head)
    state
end

"""
Non-overlapping 3D patch embedding for `(channels,frames,height,width,batch)`.
The returned token order is frame-major with width varying fastest.
"""
function patchify(model::WanTransformer, video::AbstractArray{T,5}) where T
    config = model.config
    channels, frames, height, width, batch = size(video)
    channels == config.input_channels ||
        throw(DimensionMismatch("video channel count differs"))
    pt, ph, pw = config.patch_size
    frames % pt == height % ph == width % pw == 0 ||
        throw(DimensionMismatch("video dimensions must divide patch size"))
    gf, gh, gw = frames ÷ pt, height ÷ ph, width ÷ pw
    split = reshape(video, channels, pt, gf, ph, gh, pw, gw, batch)
    # Local patch dimensions first; token axes ordered (width,height,frames).
    columns = reshape(permutedims(split, (1, 2, 4, 6, 7, 5, 3, 8)),
        channels * pt * ph * pw, gf * gh * gw * batch)
    weight = reshape(model.patch_weight, config.hidden_size, :)
    projected = reshape(weight * columns,
                        config.hidden_size, gf * gh * gw, batch)
    result = mixed_add(projected, reshape(model.patch_bias, :, 1, 1))
    (tokens=result, grid=(gf, gh, gw))
end

function sinusoidal_embedding(dimension::Integer, timesteps::AbstractVector)
    iseven(dimension) || throw(ArgumentError("sinusoidal dimension must be even"))
    half = dimension ÷ 2
    result = Matrix{Float32}(undef, dimension, length(timesteps))
    for batch in eachindex(timesteps), index in 0:half-1
        angle = Float64(timesteps[batch]) * 10000.0^(-Float64(index) / half)
        result[index + 1, batch] = Float32(cos(angle))
        result[half + index + 1, batch] = Float32(sin(angle))
    end
    result
end

function _wan_model_timesteps(timesteps::AbstractVector, training::Bool)
    scaled = 1000f0 .* Array(float32_values(timesteps))
    training ? scaled .+ 1f0 : scaled
end

function _wan_silu(x::AbstractArray)
    values = float32_values(x)
    activated = values ./ (1f0 .+ exp.(-values))
    eltype(x) === BFloat16 ? bfloat16_values(activated) :
        cast_values(eltype(x), activated)
end

function _constant_like(like, values)
    result = similar(like, eltype(like), size(values))
    if result isa CUDA.CuArray
        source = values isa CUDA.CuArray ? values : CUDA.CuArray(values)
        copyto!(result, cast_values(eltype(like), source))
    else
        copyto!(result, cast_values(eltype(like), values))
    end
    result
end

"""Undo the Wan output head patch layout into `(channels,F,H,W,batch)`."""
function unpatchify(model::WanTransformer, tokens::AbstractArray{T,3},
                    grid::NTuple{3,Int}) where T
    config = model.config
    gf, gh, gw = grid
    pt, ph, pw = config.patch_size
    expected_features = config.output_channels * pt * ph * pw
    size(tokens, 1) == expected_features ||
        throw(DimensionMismatch("head feature count differs from patch output"))
    size(tokens, 2) >= gf * gh * gw ||
        throw(DimensionMismatch("not enough tokens for output grid"))
    patches = reshape(tokens, config.output_channels, pw, ph, pt,
                      gw, gh, gf, size(tokens, 3))
    ordered = permutedims(patches, (1, 4, 7, 3, 6, 2, 5, 8))
    reshape(ordered, config.output_channels, gf * pt, gh * ph, gw * pw,
            size(tokens, 3))
end

function _wan_transformer_forward(model::WanTransformer,
                                  video::AbstractArray{T,5},
                                  timesteps::AbstractVector,
                                  text_context::AbstractArray{S,3},
                                  checkpoint_interval::Int;
                                  conditioning_video=nothing,
                                  image_features=nothing,
                                  training::Bool=false,
                                  dropout_seed::UInt64=UInt64(0)) where {T,S}
    checkpoint_interval >= 0 ||
        throw(ArgumentError("checkpoint_interval cannot be negative"))
    config = model.config
    size(video, 5) == length(timesteps) == size(text_context, 3) ||
        throw(DimensionMismatch("batch dimensions differ"))
    size(text_context, 1) == config.text_size ||
        throw(DimensionMismatch("text feature dimension differs"))
    transformer_input = if config.model_type === :i2v
        conditioning_video === nothing &&
            throw(ArgumentError("Wan I2V requires conditioning_video"))
        image_features === nothing &&
            throw(ArgumentError("Wan I2V requires image_features"))
        size(conditioning_video)[2:end] == size(video)[2:end] ||
            throw(DimensionMismatch(
                "Wan I2V conditioning video dimensions differ"))
        size(video, 1) + size(conditioning_video, 1) ==
            config.input_channels ||
            throw(DimensionMismatch(
                "Wan I2V noisy and conditioning channels do not match model"))
        cat(video, conditioning_video; dims=1)
    else
        conditioning_video === nothing ||
            throw(ArgumentError(
                "conditioning_video is only valid for Wan I2V"))
        image_features === nothing ||
            throw(ArgumentError(
                "image_features are only valid for Wan I2V"))
        video
    end
    patched = patchify(model, transformer_input)
    x = patched.tokens
    batch = size(x, 3)
    text_hidden = _projection(model.text_in, text_context;
        training=training, dropout_seed=dropout_seed, dropout_stream=1)
    context = _projection(model.text_out, gelu_tanh(text_hidden);
        training=training, dropout_seed=dropout_seed, dropout_stream=2)
    image_context_tokens = 0
    if model.image_projection !== nothing
        size(image_features, 3) == batch ||
            throw(DimensionMismatch("Wan image-feature batch differs"))
        image_context = wan_image_projection_forward(
            model.image_projection, image_features;
            training=training,
            dropout_seed=_dropout_seed(dropout_seed, 10))
        image_context_tokens = size(image_context, 2)
        context = cat(image_context, context; dims=2)
    end
    # Wan's flow interpolation uses normalized sigma in [0, 1], while the
    # pretrained DiT time embedding was trained on the 0-to-1000 scheduler
    # scale. The upstream training path additionally offsets sampled training
    # timesteps by one; inference scheduler timesteps have no offset.
    model_timesteps = _wan_model_timesteps(timesteps, training)
    time_embedding_host = sinusoidal_embedding(config.frequency_size,
                                                model_timesteps)
    time_embedding = _constant_like(model.patch_weight, time_embedding_host)
    time_hidden = _projection(model.time_in, time_embedding;
        training=training, dropout_seed=dropout_seed, dropout_stream=20)
    time_stored = _projection(model.time_out, _wan_silu(time_hidden);
        training=training, dropout_seed=dropout_seed, dropout_stream=21)
    time = float32_values(time_stored)
    time_modulation = float32_values(reshape(_projection(
        model.time_projection, _wan_silu(time_stored);
        training=training, dropout_seed=dropout_seed, dropout_stream=22),
                              config.hidden_size, 6, batch))
    grids = fill(patched.grid, batch)
    for (index, block) in enumerate(model.blocks)
        block_seed = _dropout_seed(dropout_seed, UInt64(100 + index))
        if checkpoint_interval > 0 &&
           (index - 1) % checkpoint_interval == 0
            x = Zygote.checkpointed(wan_block_forward, block, x,
                                    time_modulation, context, grids,
                                    image_context_tokens, training,
                                    block_seed)
        else
            x = wan_block_forward(block, x, time_modulation, context, grids,
                                  image_context_tokens, training, block_seed)
        end
    end
    # Official head has separate learned shift and scale offsets.
    shift = reshape(mixed_add(
        time, reshape(model.head_modulation[:, 1], :, 1)), :, 1, batch)
    scale = reshape(mixed_add(
        time, reshape(model.head_modulation[:, 2], :, 1)), :, 1, batch)
    headed = _projection(model.head, mixed_affine(
        layernorm(x; epsilon=config.epsilon), scale, shift);
        training=training, dropout_seed=dropout_seed, dropout_stream=1000)
    unpatchify(model, headed, patched.grid)
end

function wan_transformer_forward(model::WanTransformer,
                                 video::AbstractArray{T,5},
                                 timesteps::AbstractVector,
                                 text_context::AbstractArray{S,3};
                                 checkpoint_interval::Int=0,
                                 conditioning_video=nothing,
                                 image_features=nothing,
                                 training::Bool=false,
                                 dropout_seed::UInt64=UInt64(0)) where {T,S}
    _wan_transformer_forward(model, video, timesteps, text_context,
                             checkpoint_interval;
                             conditioning_video=conditioning_video,
                             image_features=image_features,
                             training=training, dropout_seed=dropout_seed)
end
