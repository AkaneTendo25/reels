struct UMT5Config
    vocab_size::Int
    hidden_size::Int
    attention_size::Int
    ffn_size::Int
    heads::Int
    layers::Int
    buckets::Int
    shared_position_bias::Bool
    epsilon::Float32
end

function UMT5Config(; vocab_size=256384, hidden_size=4096,
                    attention_size=4096, ffn_size=10240, heads=64,
                    layers=24, buckets=32, shared_position_bias=false,
                    epsilon=1f-6)
    attention_size % heads == 0 ||
        throw(ArgumentError("attention_size must be divisible by heads"))
    UMT5Config(vocab_size, hidden_size, attention_size, ffn_size, heads,
        layers, buckets, shared_position_bias, Float32(epsilon))
end

struct T5Attention
    q::DenseLayer
    k::DenseLayer
    v::DenseLayer
    o::DenseLayer
    heads::Int
end

struct T5FeedForward
    gate::DenseLayer
    fc1::DenseLayer
    fc2::DenseLayer
end

struct T5EncoderBlock
    norm1::RMSNorm
    attention::T5Attention
    norm2::RMSNorm
    feed_forward::T5FeedForward
    position_embedding::Union{Nothing,AbstractMatrix}
end

struct UMT5Encoder
    config::UMT5Config
    token_embedding::AbstractMatrix
    blocks::Vector{T5EncoderBlock}
    norm::RMSNorm
    position_embedding::Union{Nothing,AbstractMatrix}
end

function umt5_encoder_specs(config::UMT5Config=UMT5Config())
    d, a, f = config.hidden_size, config.attention_size, config.ffn_size
    specs = TensorSpec[
        TensorSpec("token_embedding.weight", [config.vocab_size, d],
            ROW_MAJOR_SOURCE),
    ]
    for block in 0:config.layers - 1
        prefix = "blocks.$block"
        push!(specs, TensorSpec("$prefix.norm1.weight", [d], VECTOR_LAYOUT))
        for projection in ("q", "k", "v", "o")
            output, input = projection == "o" ? (d, a) : (a, d)
            push!(specs, TensorSpec("$prefix.attn.$projection.weight",
                [output, input], LINEAR_OUT_IN))
        end
        push!(specs, TensorSpec("$prefix.norm2.weight", [d], VECTOR_LAYOUT))
        push!(specs, TensorSpec("$prefix.ffn.gate.0.weight",
            [f, d], LINEAR_OUT_IN))
        push!(specs, TensorSpec("$prefix.ffn.fc1.weight",
            [f, d], LINEAR_OUT_IN))
        push!(specs, TensorSpec("$prefix.ffn.fc2.weight",
            [d, f], LINEAR_OUT_IN))
        !config.shared_position_bias && push!(specs,
            TensorSpec("$prefix.pos_embedding.embedding.weight",
                [config.buckets, config.heads], ROW_MAJOR_SOURCE))
    end
    config.shared_position_bias && push!(specs,
        TensorSpec("pos_embedding.embedding.weight",
            [config.buckets, config.heads], ROW_MAJOR_SOURCE))
    push!(specs, TensorSpec("norm.weight", [d], VECTOR_LAYOUT))
    specs
end

function t5_relative_buckets(query_length::Integer, key_length::Integer,
                             bucket_count::Integer; bidirectional=true,
                             max_distance=128)
    bucket_count > 0 || throw(ArgumentError("bucket_count must be positive"))
    result = Matrix{Int}(undef, Int(query_length), Int(key_length))
    directional_buckets = bidirectional ? bucket_count ÷ 2 : bucket_count
    max_exact = directional_buckets ÷ 2
    for query in 0:Int(query_length)-1, key in 0:Int(key_length)-1
        relative = key - query
        direction = bidirectional && relative > 0 ? directional_buckets : 0
        distance = bidirectional ? abs(relative) : max(-relative, 0)
        magnitude = if distance < max_exact
            distance
        else
            scaled = max_exact + trunc(Int,
                log(Float32(distance) / Float32(max_exact)) /
                log(Float32(max_distance) / Float32(max_exact)) *
                Float32(directional_buckets - max_exact))
            min(scaled, directional_buckets - 1)
        end
        result[query + 1, key + 1] = direction + magnitude + 1
    end
    result
end

function t5_position_bias(embedding::AbstractMatrix, query_length::Integer,
                          key_length::Integer)
    # Stored as heads × buckets; result is key × query × heads × singleton batch.
    buckets = t5_relative_buckets(query_length, key_length, size(embedding, 2))
    heads = size(embedding, 1)
    bias = reshape(embedding[:, vec(buckets)], heads, query_length, key_length)
    reshape(permutedims(bias, (3, 2, 1)), key_length, query_length, heads, 1)
end

function t5_rmsnorm(norm::RMSNorm, x::AbstractArray)
    size(x, 1) == length(norm.weight) ||
        throw(DimensionMismatch("T5 norm feature dimension differs"))
    x32 = float32_values(x)
    inverse = inv.(sqrt.(sum(abs2, x32; dims=1) ./ Float32(size(x, 1)) .+
                         norm.epsilon))
    normalized = x32 .* inverse
    # Keep the normalized activation in FP32 when a low-precision frozen
    # weight is paired with an FP32 activation.  In particular, Int8 UMT5
    # promotes its residual stream to FP32 because the XXL feed-forward
    # activations can overflow Float16 even though its persistent weights are
    # stored as I8/F16.
    if eltype(x) in (Float16, BFloat16) &&
       eltype(norm.weight) in (Float16, BFloat16)
        normalized = eltype(norm.weight).(normalized)
    end
    mixed_mul(
        reshape(norm.weight, :, ntuple(_ -> 1, ndims(x) - 1)...),
        normalized)
end

function _t5_softmax(scores::AbstractArray)
    scores32 = float32_values(scores)
    maxima = maximum(scores32; dims=1)
    exponentials = exp.(scores32 .- maxima)
    converted = exponentials ./ sum(exponentials; dims=1)
    cast_values(eltype(scores), converted)
end

function t5_attention(attention::T5Attention, x::AbstractArray;
                      mask=nothing, position_bias=nothing)
    features, tokens, batch = size(x)
    projected = size(attention.q.weight, 1)
    head_size = projected ÷ attention.heads
    q = reshape(linear(attention.q, x), head_size, attention.heads, tokens, batch)
    k = reshape(linear(attention.k, x), head_size, attention.heads, tokens, batch)
    v = reshape(linear(attention.v, x), head_size, attention.heads, tokens, batch)
    scores = similar(q, eltype(q), tokens, tokens, attention.heads, batch)
    for b in 1:batch, h in 1:attention.heads
        scores[:, :, h, b] .=
            transpose(view(k, :, h, :, b)) * view(q, :, h, :, b)
    end
    position_bias === nothing || (scores = mixed_add(scores, position_bias))
    if mask !== nothing
        size(mask) == (tokens, batch) ||
            throw(DimensionMismatch("T5 mask must have shape (tokens, batch)"))
        expanded_mask = reshape(mask, tokens, 1, 1, batch)
        if eltype(scores) === BFloat16
            scores = bfloat16_values(ifelse.(
                expanded_mask, float32_values(scores), -Inf32))
        else
            scores .= ifelse.(
                expanded_mask, scores, typemin(eltype(scores)))
        end
    end
    probabilities = _t5_softmax(scores)
    attended = similar(q)
    for b in 1:batch, h in 1:attention.heads
        attended[:, h, :, b] .=
            view(v, :, h, :, b) * view(probabilities, :, :, h, b)
    end
    linear(attention.o, reshape(attended, projected, tokens, batch))
end

function t5_block_forward(block::T5EncoderBlock, x::AbstractArray;
                          mask=nothing, position_bias=nothing)
    bias = block.position_embedding === nothing ? position_bias :
        t5_position_bias(block.position_embedding, size(x, 2), size(x, 2))
    x = mixed_add(x,
        t5_attention(block.attention, t5_rmsnorm(block.norm1, x);
            mask=mask, position_bias=bias))
    hidden = t5_rmsnorm(block.norm2, x)
    gated = mixed_mul(linear(block.feed_forward.fc1, hidden),
        gelu_tanh(linear(block.feed_forward.gate, hidden)))
    mixed_add(x, linear(block.feed_forward.fc2, gated))
end

function umt5_forward(model::UMT5Encoder, ids::AbstractMatrix{<:Integer};
                      mask=nothing)
    size(ids, 1) > 0 || throw(ArgumentError("token sequence cannot be empty"))
    minimum(ids) >= 0 && maximum(ids) < size(model.token_embedding, 2) ||
        throw(ArgumentError("token id is outside the UMT5 vocabulary"))
    tokens, batch = size(ids)
    x = reshape(model.token_embedding[:, Int.(vec(ids)) .+ 1],
        size(model.token_embedding, 1), tokens, batch)
    # UMT5-XXL is not numerically safe with an FP16 residual stream: real Wan
    # prompts overflow in the gated FFN after several blocks.  QuantizedMatrix
    # still keeps the persistent dense weights in Int8; promoting activations
    # here only selects FP32 dequantization/GEMM for the current projection.
    if !isempty(model.blocks) &&
       model.blocks[1].attention.q.weight isa QuantizedMatrix &&
       eltype(x) === Float16
        x = float32_values(x)
    end
    position_bias = model.position_embedding === nothing ? nothing :
        t5_position_bias(model.position_embedding, tokens, tokens)
    for block in model.blocks
        x = t5_block_forward(block, x; mask=mask,
            position_bias=position_bias)
    end
    t5_rmsnorm(model.norm, x)
end

_umt5_compute_array(array) =
    eltype(array) === BFloat16 ? _bf16_to_float32(array) : array

_load_t5_tensor(source, spec) =
    _umt5_compute_array(load_state_tensor(source, spec))

_load_t5_dense(source, key, output, input) = DenseLayer(
    _load_t5_tensor(source,
        TensorSpec("$key.weight", [output, input], LINEAR_OUT_IN)), nothing)

function load_umt5_encoder(source::AbstractTensorSource,
                           config::UMT5Config=UMT5Config(); strict=true)
    audit = audit_state_dict(source, umt5_encoder_specs(config);
        allow_unexpected=!strict)
    _assert_clean_audit(audit)
    token_source = _load_t5_tensor(source,
        TensorSpec("token_embedding.weight",
            [config.vocab_size, config.hidden_size], ROW_MAJOR_SOURCE))
    token_embedding = permutedims(token_source)
    blocks = T5EncoderBlock[]
    for index in 0:config.layers - 1
        prefix = "blocks.$index"
        attention = T5Attention(
            _load_t5_dense(source, "$prefix.attn.q",
                config.attention_size, config.hidden_size),
            _load_t5_dense(source, "$prefix.attn.k",
                config.attention_size, config.hidden_size),
            _load_t5_dense(source, "$prefix.attn.v",
                config.attention_size, config.hidden_size),
            _load_t5_dense(source, "$prefix.attn.o",
                config.hidden_size, config.attention_size),
            config.heads)
        feed_forward = T5FeedForward(
            _load_t5_dense(source, "$prefix.ffn.gate.0",
                config.ffn_size, config.hidden_size),
            _load_t5_dense(source, "$prefix.ffn.fc1",
                config.ffn_size, config.hidden_size),
            _load_t5_dense(source, "$prefix.ffn.fc2",
                config.hidden_size, config.ffn_size))
        norm1 = RMSNorm(_load_t5_tensor(source,
            TensorSpec("$prefix.norm1.weight", [config.hidden_size],
                VECTOR_LAYOUT)), config.epsilon)
        norm2 = RMSNorm(_load_t5_tensor(source,
            TensorSpec("$prefix.norm2.weight", [config.hidden_size],
                VECTOR_LAYOUT)), config.epsilon)
        position = if config.shared_position_bias
            nothing
        else
            source_position = _load_t5_tensor(source,
                TensorSpec("$prefix.pos_embedding.embedding.weight",
                    [config.buckets, config.heads], ROW_MAJOR_SOURCE))
            permutedims(source_position)
        end
        push!(blocks, T5EncoderBlock(norm1, attention, norm2, feed_forward,
            position))
    end
    shared_position = if config.shared_position_bias
        source_position = _load_t5_tensor(source,
            TensorSpec("pos_embedding.embedding.weight",
                [config.buckets, config.heads], ROW_MAJOR_SOURCE))
        permutedims(source_position)
    else
        nothing
    end
    final_norm = RMSNorm(_load_t5_tensor(source,
        TensorSpec("norm.weight", [config.hidden_size], VECTOR_LAYOUT)),
        config.epsilon)
    UMT5Encoder(config, token_embedding, blocks, final_norm, shared_position)
end

load_umt5_encoder(path::AbstractString, config::UMT5Config=UMT5Config();
                   kwargs...) =
    load_umt5_encoder(open_tensor_source(path), config; kwargs...)

const UMT5_INT8_FORMAT = "reels-umt5-int8-v1"

_umt5_dense_keys(config::UMT5Config) = String[
    key for block in 0:config.layers - 1
    for key in (
        "blocks.$block.attn.q.weight",
        "blocks.$block.attn.k.weight",
        "blocks.$block.attn.v.weight",
        "blocks.$block.attn.o.weight",
        "blocks.$block.ffn.gate.0.weight",
        "blocks.$block.ffn.fc1.weight",
        "blocks.$block.ffn.fc2.weight",
    )
]

"""Return a portable SafeTensors state dictionary for an Int8 UMT5 encoder."""
function quantized_umt5_state_dict(model::UMT5Encoder;
                                   compute_type::Type=Float16)
    compute_type in (Float16, Float32) || throw(ArgumentError(
        "quantized UMT5 compute_type must be Float16 or Float32"))
    state = umt5_encoder_state_dict(model)
    for key in _umt5_dense_keys(model.config)
        quantized = quantize_frozen_matrix(state[key], compute_type)
        state[key] = Array(quantized.values)
        state["$key.scale"] = Array(quantized.scales)
    end
    for (key, value) in state
        eltype(value) <: AbstractFloat || continue
        endswith(key, ".scale") && continue
        state[key] = compute_type.(value)
    end
    state
end

"""Write UMT5 weights using per-output-channel symmetric Int8 matrices."""
function write_quantized_umt5(path::AbstractString, model::UMT5Encoder;
                              compute_type::Type=Float16,
                              metadata=Dict{String,String}())
    info = Dict{String,String}(metadata)
    info["format"] = UMT5_INT8_FORMAT
    info["quantization"] = "int8-per-output-channel"
    info["compute_dtype"] = compute_type === Float16 ? "float16" : "float32"
    write_safetensors(path,
        quantized_umt5_state_dict(model; compute_type=compute_type);
        metadata=info)
end

function _load_quantized_t5_dense(source, key, output, input,
                                  ::Type{T}) where T
    values = load_state_tensor(source,
        TensorSpec("$key.weight", [output, input], LINEAR_OUT_IN))
    eltype(values) === Int8 || throw(ArgumentError(
        "$key.weight must use SafeTensors I8 storage"))
    scales = load_state_tensor(source,
        TensorSpec("$key.weight.scale", [output], VECTOR_LAYOUT))
    DenseLayer(QuantizedMatrix{T,typeof(values),Vector{Float32}}(
        values, Float32.(scales)), nothing)
end

"""Return the deterministic tensor schema for an Int8 UMT5 checkpoint."""
function quantized_umt5_encoder_specs(config::UMT5Config=UMT5Config())
    dense = Set(_umt5_dense_keys(config))
    specs = TensorSpec[]
    for spec in umt5_encoder_specs(config)
        push!(specs, spec)
        spec.source_key in dense || continue
        push!(specs, TensorSpec("$(spec.source_key).scale",
            [spec.source_shape[1]], VECTOR_LAYOUT))
    end
    specs
end

"""Load a `reels-umt5-int8-v1` SafeTensors checkpoint without requantizing."""
function load_quantized_umt5_encoder(source::AbstractTensorSource,
                                     config::UMT5Config=UMT5Config();
                                     compute_type::Type=Float16, strict=true)
    compute_type in (Float16, Float32) || throw(ArgumentError(
        "quantized UMT5 compute_type must be Float16 or Float32"))
    audit = audit_state_dict(source, quantized_umt5_encoder_specs(config);
        allow_unexpected=!strict)
    _assert_clean_audit(audit)
    token_source = _load_t5_tensor(source,
        TensorSpec("token_embedding.weight",
            [config.vocab_size, config.hidden_size], ROW_MAJOR_SOURCE))
    blocks = T5EncoderBlock[]
    for index in 0:config.layers - 1
        prefix = "blocks.$index"
        attention = T5Attention(
            _load_quantized_t5_dense(source, "$prefix.attn.q",
                config.attention_size, config.hidden_size, compute_type),
            _load_quantized_t5_dense(source, "$prefix.attn.k",
                config.attention_size, config.hidden_size, compute_type),
            _load_quantized_t5_dense(source, "$prefix.attn.v",
                config.attention_size, config.hidden_size, compute_type),
            _load_quantized_t5_dense(source, "$prefix.attn.o",
                config.hidden_size, config.attention_size, compute_type),
            config.heads)
        feed_forward = T5FeedForward(
            _load_quantized_t5_dense(source, "$prefix.ffn.gate.0",
                config.ffn_size, config.hidden_size, compute_type),
            _load_quantized_t5_dense(source, "$prefix.ffn.fc1",
                config.ffn_size, config.hidden_size, compute_type),
            _load_quantized_t5_dense(source, "$prefix.ffn.fc2",
                config.hidden_size, config.ffn_size, compute_type))
        norm1 = RMSNorm(compute_type.(_load_t5_tensor(source,
            TensorSpec("$prefix.norm1.weight", [config.hidden_size],
                VECTOR_LAYOUT))), config.epsilon)
        norm2 = RMSNorm(compute_type.(_load_t5_tensor(source,
            TensorSpec("$prefix.norm2.weight", [config.hidden_size],
                VECTOR_LAYOUT))), config.epsilon)
        position = config.shared_position_bias ? nothing : permutedims(
            compute_type.(_load_t5_tensor(source,
                TensorSpec("$prefix.pos_embedding.embedding.weight",
                    [config.buckets, config.heads], ROW_MAJOR_SOURCE))))
        push!(blocks, T5EncoderBlock(norm1, attention, norm2, feed_forward,
            position))
    end
    shared_position = config.shared_position_bias ? permutedims(
        compute_type.(_load_t5_tensor(source,
            TensorSpec("pos_embedding.embedding.weight",
                [config.buckets, config.heads], ROW_MAJOR_SOURCE)))) : nothing
    final_norm = RMSNorm(compute_type.(_load_t5_tensor(source,
        TensorSpec("norm.weight", [config.hidden_size], VECTOR_LAYOUT))),
        config.epsilon)
    UMT5Encoder(config, permutedims(compute_type.(token_source)), blocks,
        final_norm, shared_position)
end

load_quantized_umt5_encoder(path::AbstractString,
                            config::UMT5Config=UMT5Config(); kwargs...) =
    load_quantized_umt5_encoder(open_tensor_source(path), config; kwargs...)

function move_to_device(model::UMT5Encoder, transfer)
    move_attention(attention) = T5Attention(
        _move_dense(attention.q, transfer), _move_dense(attention.k, transfer),
        _move_dense(attention.v, transfer), _move_dense(attention.o, transfer),
        attention.heads)
    move_feed_forward(ffn) = T5FeedForward(
        _move_dense(ffn.gate, transfer), _move_dense(ffn.fc1, transfer),
        _move_dense(ffn.fc2, transfer))
    move_block(block) = T5EncoderBlock(
        _move_norm(block.norm1, transfer), move_attention(block.attention),
        _move_norm(block.norm2, transfer), move_feed_forward(block.feed_forward),
        _move_array(block.position_embedding, transfer))
    UMT5Encoder(model.config, transfer(model.token_embedding),
        [move_block(block) for block in model.blocks],
        _move_norm(model.norm, transfer),
        _move_array(model.position_embedding, transfer))
end

move_to_device(model::UMT5Encoder, ::Val{:cpu}) =
    move_to_device(model, Array)
function move_to_device(model::UMT5Encoder, ::Val{:cuda})
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    move_to_device(model, CUDA.CuArray)
end
move_to_device(model::UMT5Encoder, device::Symbol) =
    device === :cpu ? move_to_device(model, Val(:cpu)) :
    device === :cuda ? move_to_device(model, Val(:cuda)) :
    throw(ArgumentError("unsupported device: $device"))
move_to_device(model::UMT5Encoder, device::Symbol, precision::Symbol) =
    move_to_device(model, array_transfer(device, precision))

function umt5_encoder_state_dict(model::UMT5Encoder)
    state = Dict{String,AbstractArray}()
    state["token_embedding.weight"] = permutedims(Array(model.token_embedding))
    for (offset, block) in enumerate(model.blocks)
        prefix = "blocks.$(offset - 1)"
        state["$prefix.norm1.weight"] = Array(block.norm1.weight)
        for (name, layer) in (
            "q" => block.attention.q, "k" => block.attention.k,
            "v" => block.attention.v, "o" => block.attention.o)
            state["$prefix.attn.$name.weight"] = Array(layer.weight)
        end
        state["$prefix.norm2.weight"] = Array(block.norm2.weight)
        state["$prefix.ffn.gate.0.weight"] =
            Array(block.feed_forward.gate.weight)
        state["$prefix.ffn.fc1.weight"] = Array(block.feed_forward.fc1.weight)
        state["$prefix.ffn.fc2.weight"] = Array(block.feed_forward.fc2.weight)
        block.position_embedding === nothing ||
            (state["$prefix.pos_embedding.embedding.weight"] =
                permutedims(Array(block.position_embedding)))
    end
    model.position_embedding === nothing ||
        (state["pos_embedding.embedding.weight"] =
            permutedims(Array(model.position_embedding)))
    state["norm.weight"] = Array(model.norm.weight)
    state
end
