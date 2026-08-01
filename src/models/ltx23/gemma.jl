# Native Gemma-3 text encoder for LTX-2.3 preprocessing.
# Architecture follows the official Gemma-3 implementation.

struct Gemma3TextConfig
    vocab_size::Int
    hidden_size::Int
    intermediate_size::Int
    layers::Int
    heads::Int
    kv_heads::Int
    head_dim::Int
    epsilon::Float32
    rope_theta::Float32
    local_rope_theta::Float32
    rope_scaling_factor::Float32
    sliding_window::Int
    sliding_window_pattern::Int
    query_pre_attention_scalar::Float32
    max_length::Int
end

function Gemma3TextConfig(; vocab_size=262208, hidden_size=3840,
                          intermediate_size=15360, layers=48,
                          heads=16, kv_heads=8, head_dim=256,
                          epsilon=1f-6, rope_theta=1_000_000f0,
                          local_rope_theta=10_000f0,
                          rope_scaling_factor=8f0,
                          sliding_window=1024,
                          sliding_window_pattern=6,
                          query_pre_attention_scalar=256f0,
                          max_length=1024)
    all(>(0), (vocab_size, hidden_size, intermediate_size, layers,
               heads, kv_heads, head_dim, sliding_window,
               sliding_window_pattern, max_length)) ||
        throw(ArgumentError("Gemma dimensions must be positive"))
    heads % kv_heads == 0 ||
        throw(ArgumentError("Gemma query heads must divide key/value heads"))
    Gemma3TextConfig(Int(vocab_size), Int(hidden_size),
        Int(intermediate_size), Int(layers), Int(heads), Int(kv_heads),
        Int(head_dim), Float32(epsilon), Float32(rope_theta),
        Float32(local_rope_theta), Float32(rope_scaling_factor),
        Int(sliding_window), Int(sliding_window_pattern),
        Float32(query_pre_attention_scalar), Int(max_length))
end

function gemma3_text_config(values::AbstractDict)
    text = get(values, "text_config", values)
    scaling = _ltx_config_value(text, "rope_scaling",
                                Dict("factor" => 8))
    Gemma3TextConfig(
        vocab_size=Int(_ltx_config_value(text, "vocab_size", 262208)),
        hidden_size=Int(_ltx_config_value(text, "hidden_size", 3840)),
        intermediate_size=Int(_ltx_config_value(
            text, "intermediate_size", 15360)),
        layers=Int(_ltx_config_value(text, "num_hidden_layers", 48)),
        heads=Int(_ltx_config_value(text, "num_attention_heads", 16)),
        kv_heads=Int(_ltx_config_value(text, "num_key_value_heads", 8)),
        head_dim=Int(_ltx_config_value(text, "head_dim", 256)),
        epsilon=Float32(_ltx_config_value(text, "rms_norm_eps", 1e-6)),
        rope_theta=Float32(_ltx_config_value(text, "rope_theta", 1_000_000)),
        local_rope_theta=Float32(_ltx_config_value(
            text, "rope_local_base_freq", 10_000)),
        rope_scaling_factor=Float32(_ltx_config_value(
            scaling, "factor", 8)),
        sliding_window=Int(_ltx_config_value(text, "sliding_window", 1024)),
        sliding_window_pattern=Int(_ltx_config_value(
            text, "sliding_window_pattern", 6)),
        query_pre_attention_scalar=Float32(_ltx_config_value(
            text, "query_pre_attn_scalar", 256)),
        max_length=Int(_ltx_config_value(text,
            "ltx_max_length", min(1024, Int(_ltx_config_value(
                text, "max_position_embeddings", 131072))))))
end

function gemma3_text_config(path::AbstractString)
    config_path = isdir(path) ? joinpath(path, "config.json") : String(path)
    isfile(config_path) ||
        throw(ArgumentError("Gemma config does not exist: $config_path"))
    values = parse_json(read(config_path, String))
    values isa AbstractDict ||
        throw(ArgumentError("Gemma configuration root must be an object"))
    gemma3_text_config(values)
end

function gemma3_text_encoder_specs(config::Gemma3TextConfig=
        Gemma3TextConfig(); prefix="language_model.model.")
    specs = TensorSpec[
        TensorSpec(prefix * "embed_tokens.weight",
                   [config.vocab_size, config.hidden_size],
                   ROW_MAJOR_SOURCE)
    ]
    for layer in 0:config.layers-1
        root = prefix * "layers.$layer"
        push!(specs,
            TensorSpec("$root.input_layernorm.weight",
                       [config.hidden_size], VECTOR_LAYOUT),
            TensorSpec("$root.self_attn.q_proj.weight",
                       [config.heads * config.head_dim,
                        config.hidden_size], LINEAR_OUT_IN),
            TensorSpec("$root.self_attn.k_proj.weight",
                       [config.kv_heads * config.head_dim,
                        config.hidden_size], LINEAR_OUT_IN),
            TensorSpec("$root.self_attn.v_proj.weight",
                       [config.kv_heads * config.head_dim,
                        config.hidden_size], LINEAR_OUT_IN),
            TensorSpec("$root.self_attn.q_norm.weight",
                       [config.head_dim], VECTOR_LAYOUT),
            TensorSpec("$root.self_attn.k_norm.weight",
                       [config.head_dim], VECTOR_LAYOUT),
            TensorSpec("$root.self_attn.o_proj.weight",
                       [config.hidden_size,
                        config.heads * config.head_dim], LINEAR_OUT_IN),
            TensorSpec("$root.post_attention_layernorm.weight",
                       [config.hidden_size], VECTOR_LAYOUT),
            TensorSpec("$root.pre_feedforward_layernorm.weight",
                       [config.hidden_size], VECTOR_LAYOUT),
            TensorSpec("$root.mlp.gate_proj.weight",
                       [config.intermediate_size,
                        config.hidden_size], LINEAR_OUT_IN),
            TensorSpec("$root.mlp.up_proj.weight",
                       [config.intermediate_size,
                        config.hidden_size], LINEAR_OUT_IN),
            TensorSpec("$root.mlp.down_proj.weight",
                       [config.hidden_size,
                        config.intermediate_size], LINEAR_OUT_IN),
            TensorSpec("$root.post_feedforward_layernorm.weight",
                       [config.hidden_size], VECTOR_LAYOUT))
    end
    push!(specs, TensorSpec(prefix * "norm.weight",
                            [config.hidden_size], VECTOR_LAYOUT))
    specs
end

struct Gemma3Norm{W}
    weight::W
    epsilon::Float32
end

struct Gemma3Attention{Q,K,V,O,QN,KN}
    q::Q
    k::K
    v::V
    o::O
    q_norm::QN
    k_norm::KN
end

struct Gemma3MLP{G,U,D}
    gate::G
    up::U
    down::D
end

struct Gemma3DecoderBlock{N,A,P,F,M,O}
    input_norm::N
    attention::A
    post_attention_norm::P
    pre_ffn_norm::F
    mlp::M
    post_ffn_norm::O
end

struct Gemma3TextEncoder{E,B,N}
    config::Gemma3TextConfig
    embedding::E
    blocks::B
    final_norm::N
end

function _gemma_dense(rng, output, input; initialize=true)
    weight = initialize ?
        randn(rng, Float32, output, input) .* 0.02f0 :
        zeros(Float32, output, input)
    DenseLayer(weight, zeros(Float32, output))
end

_gemma_norm(config; initialize=true) =
    Gemma3Norm(zeros(Float32, config.hidden_size), config.epsilon)
_gemma_head_norm(config; initialize=true) =
    Gemma3Norm(zeros(Float32, config.head_dim), config.epsilon)

function Gemma3TextEncoder(config::Gemma3TextConfig=Gemma3TextConfig();
                           rng=Random.default_rng(), initialize=true)
    embedding = initialize ?
        randn(rng, Float32, config.hidden_size, config.vocab_size) .* 0.02f0 :
        zeros(Float32, config.hidden_size, config.vocab_size)
    blocks = [Gemma3DecoderBlock(
        _gemma_norm(config; initialize=initialize),
        Gemma3Attention(
            _gemma_dense(rng, config.heads * config.head_dim,
                         config.hidden_size; initialize=initialize),
            _gemma_dense(rng, config.kv_heads * config.head_dim,
                         config.hidden_size; initialize=initialize),
            _gemma_dense(rng, config.kv_heads * config.head_dim,
                         config.hidden_size; initialize=initialize),
            _gemma_dense(rng, config.hidden_size,
                         config.heads * config.head_dim;
                         initialize=initialize),
            _gemma_head_norm(config; initialize=initialize),
            _gemma_head_norm(config; initialize=initialize)),
        _gemma_norm(config; initialize=initialize),
        _gemma_norm(config; initialize=initialize),
        Gemma3MLP(
            _gemma_dense(rng, config.intermediate_size,
                         config.hidden_size; initialize=initialize),
            _gemma_dense(rng, config.intermediate_size,
                         config.hidden_size; initialize=initialize),
            _gemma_dense(rng, config.hidden_size,
                         config.intermediate_size; initialize=initialize)),
        _gemma_norm(config; initialize=initialize))
        for _ in 1:config.layers]
    Gemma3TextEncoder(config, embedding, blocks,
                      _gemma_norm(config; initialize=initialize))
end

function gemma3_norm(norm::Gemma3Norm, x)
    size(x, 1) == length(norm.weight) ||
        throw(DimensionMismatch("Gemma RMSNorm feature dimension differs"))
    x32 = float32_values(x)
    normalized = x32 ./ sqrt.(sum(abs2, x32; dims=1) ./
                              Float32(size(x, 1)) .+ norm.epsilon)
    weighted = normalized .* reshape(
        1f0 .+ float32_values(norm.weight), :,
        ntuple(_ -> 1, ndims(x) - 1)...)
    cast_values(eltype(x), weighted)
end

function _gemma_rope(config::Gemma3TextConfig, tokens, like;
                     global_layer=false)
    theta = global_layer ? config.rope_theta : config.local_rope_theta
    scale = global_layer ? config.rope_scaling_factor : 1f0
    half = config.head_dim ÷ 2
    angles = Array{Float32}(undef, half, tokens, 1)
    for token in 1:tokens, index in 0:half-1
        inv_frequency = theta ^ (-2f0 * Float32(index) /
                                 Float32(config.head_dim))
        angles[index + 1, token, 1] =
            Float32(token - 1) * inv_frequency / scale
    end
    cosine = reshape(cos.(angles), half, 1, tokens, 1)
    sine = reshape(sin.(angles), half, 1, tokens, 1)
    transfer(values) = begin
        source = like isa CUDA.CuArray ? CUDA.CuArray(values) : values
        cast_values(eltype(like), source)
    end
    (transfer(cosine), transfer(sine))
end

function _gemma_attention_mask(mask::AbstractMatrix, batch::Integer;
                               sliding_window=nothing)
    tokens = size(mask, 1)
    visible = falses(tokens, tokens)
    for query in 1:tokens, key in 1:tokens
        allowed = mask[key, batch] && key <= query
        sliding_window === nothing ||
            (allowed &= query - key < sliding_window)
        visible[key, query] = allowed
    end
    # Fully masked padding queries are discarded by the downstream feature
    # mask, but must still have a finite attention row.
    for query in 1:tokens
        any(view(visible, :, query)) || (visible[query, query] = true)
    end
    visible
end

function gemma3_attention_forward(attention::Gemma3Attention,
                                  config::Gemma3TextConfig, x,
                                  mask::AbstractMatrix;
                                  global_layer=false)
    q = attention.q(x)
    k = attention.k(x)
    v = attention.v(x)
    tokens, batch = size(x, 2), size(x, 3)
    q = reshape(q, config.head_dim, config.heads, tokens, batch)
    k = reshape(k, config.head_dim, config.kv_heads, tokens, batch)
    v = reshape(v, config.head_dim, config.kv_heads, tokens, batch)
    q = gemma3_norm(attention.q_norm, q)
    k = gemma3_norm(attention.k_norm, k)
    frequencies = _gemma_rope(config, tokens, q;
                              global_layer=global_layer)
    q = ltx_apply_rope(q, frequencies)
    k = ltx_apply_rope(k, frequencies)
    groups = config.heads ÷ config.kv_heads
    k = groups == 1 ? k : repeat(k; inner=(1, groups, 1, 1))
    v = groups == 1 ? v : repeat(v; inner=(1, groups, 1, 1))
    attended = similar(q)
    window = global_layer ? nothing : config.sliding_window
    for index in 1:batch
        visible = _gemma_attention_mask(mask, index;
                                        sliding_window=window)
        result = memory_efficient_attention(
            view(q, :, :, :, index:index),
            view(k, :, :, :, index:index),
            view(v, :, :, :, index:index); mask=visible).output
        attended[:, :, :, index:index] .= result
    end
    attention.o(reshape(attended,
        config.heads * config.head_dim, tokens, batch))
end

function gemma3_block_forward(block::Gemma3DecoderBlock,
                              config::Gemma3TextConfig, x,
                              mask::AbstractMatrix, layer_index::Integer)
    global_layer = layer_index % config.sliding_window_pattern == 0
    attended = gemma3_attention_forward(block.attention, config,
        gemma3_norm(block.input_norm, x), mask;
        global_layer=global_layer)
    x = mixed_add(x, gemma3_norm(block.post_attention_norm, attended))
    normalized = gemma3_norm(block.pre_ffn_norm, x)
    gated = mixed_mul(gelu_tanh(block.mlp.gate(normalized)),
                      block.mlp.up(normalized))
    fed = block.mlp.down(gated)
    mixed_add(x, gemma3_norm(block.post_ffn_norm, fed))
end

"""
Run Gemma-3 and return the 49 hidden-state taps consumed by LTX-2.3.
Inputs use zero-based token IDs with `(tokens,batch)` layout.
"""
function gemma3_forward(model::Gemma3TextEncoder,
                        ids::AbstractMatrix{<:Integer},
                        mask::AbstractMatrix)
    size(ids) == size(mask) ||
        throw(DimensionMismatch("Gemma token IDs and mask shapes differ"))
    size(ids, 1) <= model.config.max_length ||
        throw(DimensionMismatch("Gemma sequence exceeds configured maximum"))
    minimum(ids) >= 0 && maximum(ids) < model.config.vocab_size ||
        throw(ArgumentError("Gemma token ID is outside vocabulary"))
    tokens, batch = size(ids)
    # A single gather keeps token lookup GPU-safe; scalar CuArray indexing is
    # deliberately disabled by CUDA.jl.
    indices = vec(Int.(ids)) .+ 1
    hidden = reshape(model.embedding[:, indices],
                     model.config.hidden_size, tokens, batch)
    scale = sqrt(Float32(model.config.hidden_size))
    hidden = cast_values(eltype(hidden), float32_values(hidden) .* scale)
    states = Vector{typeof(hidden)}()
    push!(states, copy(hidden))
    for (index, block) in enumerate(model.blocks)
        hidden = gemma3_block_forward(block, model.config,
                                      hidden, mask, index)
        index < length(model.blocks) && push!(states, copy(hidden))
    end
    hidden = gemma3_norm(model.final_norm, hidden)
    push!(states, hidden)
    states
end

function _load_gemma_dense!(source, key, layer)
    loaded = load_state_tensor(source,
        TensorSpec(key, collect(size(layer.weight)), LINEAR_OUT_IN))
    copyto!(layer.weight, _umt5_compute_array(loaded))
    fill!(layer.bias, zero(eltype(layer.bias)))
    layer
end

function _load_gemma_norm!(source, key, norm)
    copyto!(norm.weight, _umt5_compute_array(load_state_tensor(source,
        TensorSpec(key, [length(norm.weight)], VECTOR_LAYOUT))))
    norm
end

function load_gemma3_text_encoder(source::AbstractTensorSource,
                                  config::Gemma3TextConfig=Gemma3TextConfig();
                                  strict=false,
                                  prefix="language_model.model.")
    specs = gemma3_text_encoder_specs(config; prefix=prefix)
    audit = audit_state_dict(source, specs; allow_unexpected=!strict)
    isempty(audit.missing) && isempty(audit.shape_mismatches) ||
        _assert_clean_audit(StateDictAudit(audit.missing, String[],
                                          audit.shape_mismatches))
    model = Gemma3TextEncoder(config; rng=Xoshiro(0), initialize=false)
    embedding = load_state_tensor(source,
        TensorSpec(prefix * "embed_tokens.weight",
            [config.vocab_size, config.hidden_size], ROW_MAJOR_SOURCE))
    copyto!(model.embedding, permutedims(_umt5_compute_array(embedding)))
    for (index, block) in enumerate(model.blocks)
        root = prefix * "layers.$(index - 1)"
        _load_gemma_norm!(source, "$root.input_layernorm.weight",
                          block.input_norm)
        _load_gemma_dense!(source, "$root.self_attn.q_proj.weight",
                           block.attention.q)
        _load_gemma_dense!(source, "$root.self_attn.k_proj.weight",
                           block.attention.k)
        _load_gemma_dense!(source, "$root.self_attn.v_proj.weight",
                           block.attention.v)
        _load_gemma_norm!(source, "$root.self_attn.q_norm.weight",
                          block.attention.q_norm)
        _load_gemma_norm!(source, "$root.self_attn.k_norm.weight",
                          block.attention.k_norm)
        _load_gemma_dense!(source, "$root.self_attn.o_proj.weight",
                           block.attention.o)
        _load_gemma_norm!(source, "$root.post_attention_layernorm.weight",
                          block.post_attention_norm)
        _load_gemma_norm!(source, "$root.pre_feedforward_layernorm.weight",
                          block.pre_ffn_norm)
        _load_gemma_dense!(source, "$root.mlp.gate_proj.weight",
                           block.mlp.gate)
        _load_gemma_dense!(source, "$root.mlp.up_proj.weight",
                           block.mlp.up)
        _load_gemma_dense!(source, "$root.mlp.down_proj.weight",
                           block.mlp.down)
        _load_gemma_norm!(source, "$root.post_feedforward_layernorm.weight",
                          block.post_ffn_norm)
    end
    _load_gemma_norm!(source, prefix * "norm.weight", model.final_norm)
    model
end

function _gemma_index_path(path::AbstractString)
    isdir(path) || return String(path)
    candidates = filter(name -> endswith(name, ".safetensors.index.json"),
                        readdir(path; join=true))
    length(candidates) == 1 ||
        throw(ArgumentError("Gemma directory must contain one SafeTensors index"))
    only(candidates)
end

load_gemma3_text_encoder(path::AbstractString;
                         config=gemma3_text_config(
                             isdir(path) ? path : dirname(path)), kwargs...) =
    load_gemma3_text_encoder(open_tensor_source(_gemma_index_path(path)),
                             config; kwargs...)

_move_gemma_norm(norm, transfer) =
    Gemma3Norm(transfer(norm.weight), norm.epsilon)

function move_to_device(model::Gemma3TextEncoder, transfer)
    blocks = [Gemma3DecoderBlock(
        _move_gemma_norm(block.input_norm, transfer),
        Gemma3Attention(
            _move_dense(block.attention.q, transfer),
            _move_dense(block.attention.k, transfer),
            _move_dense(block.attention.v, transfer),
            _move_dense(block.attention.o, transfer),
            _move_gemma_norm(block.attention.q_norm, transfer),
            _move_gemma_norm(block.attention.k_norm, transfer)),
        _move_gemma_norm(block.post_attention_norm, transfer),
        _move_gemma_norm(block.pre_ffn_norm, transfer),
        Gemma3MLP(_move_dense(block.mlp.gate, transfer),
                  _move_dense(block.mlp.up, transfer),
                  _move_dense(block.mlp.down, transfer)),
        _move_gemma_norm(block.post_ffn_norm, transfer))
        for block in model.blocks]
    Gemma3TextEncoder(model.config, transfer(model.embedding), blocks,
                      _move_gemma_norm(model.final_norm, transfer))
end

move_to_device(model::Gemma3TextEncoder, device::Symbol,
               precision::Symbol=:fp32) =
    move_to_device(model, array_transfer(device, precision))
