# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

struct LTXTextConnectorConfig
    gemma_hidden_size::Int
    gemma_hidden_layers::Int
    output_dim::Int
    heads::Int
    head_dim::Int
    layers::Int
    registers::Int
    max_position::Int
    rope_theta::Float32
    gated_attention::Bool
    epsilon::Float32
end

function LTXTextConnectorConfig(; gemma_hidden_size=3840,
                                gemma_hidden_layers=49,
                                output_dim=4096, heads=32,
                                head_dim=128, layers=8,
                                registers=128, max_position=4096,
                                rope_theta=10_000f0,
                                gated_attention=true, epsilon=1f-6)
    gemma_hidden_size > 0 && gemma_hidden_layers > 0 &&
        output_dim > 0 && heads > 0 && head_dim > 0 && layers > 0 &&
        registers >= 0 && max_position > 0 ||
        throw(ArgumentError("LTX text connector dimensions are invalid"))
    output_dim == heads * head_dim ||
        throw(ArgumentError("LTX connector output must equal heads × head dimension"))
    LTXTextConnectorConfig(Int(gemma_hidden_size),
        Int(gemma_hidden_layers), Int(output_dim), Int(heads),
        Int(head_dim), Int(layers), Int(registers), Int(max_position),
        Float32(rope_theta), Bool(gated_attention), Float32(epsilon))
end

function ltx23_text_connector_config(values::AbstractDict)
    transformer = get(values, "transformer", values)
    maxima = _ltx_config_value(transformer,
        "connector_positional_embedding_max_pos", [4096])
    length(maxima) == 1 ||
        throw(ArgumentError("LTX connector position limit must have one entry"))
    LTXTextConnectorConfig(
        output_dim=Int(_ltx_config_value(transformer,
            "num_attention_heads", 32)) *
            Int(_ltx_config_value(transformer, "attention_head_dim", 128)),
        heads=Int(_ltx_config_value(transformer,
            "connector_num_attention_heads", 32)),
        head_dim=Int(_ltx_config_value(transformer,
            "connector_attention_head_dim", 128)),
        layers=Int(_ltx_config_value(transformer,
            "connector_num_layers", 8)),
        registers=Int(_ltx_config_value(transformer,
            "connector_num_learnable_registers", 128)),
        max_position=Int(only(maxima)),
        rope_theta=Float32(_ltx_config_value(transformer,
            "positional_embedding_theta", 10_000)),
        gated_attention=Bool(_ltx_config_value(transformer,
            "connector_apply_gated_attention", true)))
end

function ltx23_text_connector_specs(
        config::LTXTextConnectorConfig=LTXTextConnectorConfig();
        projection_prefix="text_embedding_projection.video_aggregate_embed",
        connector_prefix="model.diffusion_model.video_embeddings_connector")
    specs = TensorSpec[]
    flat = config.gemma_hidden_size * config.gemma_hidden_layers
    _ltx_linear_specs!(specs, projection_prefix, config.output_dim, flat)
    config.registers > 0 && push!(specs,
        TensorSpec("$connector_prefix.learnable_registers",
                   [config.registers, config.output_dim], ROW_MAJOR_SOURCE))
    for layer in 0:config.layers-1
        root = "$connector_prefix.transformer_1d_blocks.$layer"
        _ltx_attention_specs!(specs, "$root.attn1",
            config.output_dim, config.output_dim, config.output_dim;
            gated=config.gated_attention, heads=config.heads)
        _ltx_ff_specs!(specs, "$root.ff", config.output_dim)
    end
    specs
end

struct LTXTextConnectorBlock{A,I,O}
    attention::A
    ffn_in::I
    ffn_out::O
end

struct LTXTextConnector{P,B,R}
    config::LTXTextConnectorConfig
    projection::P
    blocks::B
    registers::R
end

struct LTXTextConditioner{T,G,C}
    tokenizer::T
    gemma::G
    connector::C
end

struct LTXTextEncoding{C,M}
    context::C
    mask::M
    lengths::Vector{Int}
end

Base.close(conditioner::LTXTextConditioner) = close(conditioner.tokenizer)

function _ltx_connector_attention(config::LTXTextConnectorConfig;
                                  rng=Random.default_rng(), initialize=true)
    d = config.output_dim
    LTXAttention(
        _ltx_dense(rng, d, d; initialize=initialize),
        _ltx_dense(rng, d, d; initialize=initialize),
        _ltx_dense(rng, d, d; initialize=initialize),
        _ltx_dense(rng, d, d; initialize=initialize),
        RMSNorm(ones(Float32, d), config.epsilon),
        RMSNorm(ones(Float32, d), config.epsilon),
        config.gated_attention ?
            _ltx_dense(rng, config.heads, d; initialize=initialize) :
            nothing,
        config.heads, config.head_dim)
end

function LTXTextConnector(config::LTXTextConnectorConfig=
        LTXTextConnectorConfig(); rng=Random.default_rng(), initialize=true)
    d = config.output_dim
    projection = _ltx_dense(rng, d,
        config.gemma_hidden_size * config.gemma_hidden_layers;
        initialize=initialize)
    blocks = [LTXTextConnectorBlock(
        _ltx_connector_attention(config; rng=rng, initialize=initialize),
        _ltx_dense(rng, 4d, d; initialize=initialize),
        _ltx_dense(rng, d, 4d; initialize=initialize))
        for _ in 1:config.layers]
    registers = config.registers == 0 ? nothing :
        (initialize ? 2f0 .* rand(rng, Float32, d, config.registers) .- 1f0 :
                      zeros(Float32, d, config.registers))
    LTXTextConnector(config, projection, blocks, registers)
end

"""
Per-token, per-layer RMS normalization and flattening used by LTX-2.3's
Gemma V2 feature extractor. Hidden states use `(features,tokens,batch)`.
"""
function ltx23_gemma_features(hidden_states::AbstractVector,
                              mask::AbstractMatrix,
                              config::LTXTextConnectorConfig)
    length(hidden_states) == config.gemma_hidden_layers ||
        throw(DimensionMismatch("Gemma hidden-state layer count differs"))
    tokens, batch = size(mask)
    all(state -> size(state) ==
        (config.gemma_hidden_size, tokens, batch), hidden_states) ||
        throw(DimensionMismatch("Gemma hidden-state shapes differ"))
    normalized = map(hidden_states) do state
        state32 = float32_values(state)
        value = state32 ./ sqrt.(sum(abs2, state32; dims=1) ./
                                 Float32(config.gemma_hidden_size) .+ 1f-6)
        host_mask = reshape(Float32.(mask), 1, tokens, batch)
        device_mask = state isa CUDA.CuArray ?
            CUDA.CuArray(host_mask) : host_mask
        cast_values(eltype(state), value .* device_mask)
    end
    stacked = cat(normalized...; dims=4) # (D,T,B,L)
    flattened = reshape(permutedims(stacked, (4, 1, 2, 3)),
        config.gemma_hidden_layers * config.gemma_hidden_size,
        tokens, batch)
    scale = Float32(sqrt(config.output_dim / config.gemma_hidden_size))
    cast_values(eltype(flattened), float32_values(flattened) .* scale)
end

function ltx23_text_conditioner_forward(conditioner::LTXTextConditioner,
                                        tokenized::TokenizedText)
    states = gemma3_forward(conditioner.gemma,
                            tokenized.ids, tokenized.mask)
    context = ltx23_text_connector_forward(
        conditioner.connector, states, tokenized.mask)
    all(isfinite, context) || throw(ArgumentError(
        "LTX text encoder produced non-finite context; use BF16 or FP32 " *
        "instead of FP16 for Gemma-3 conditioning"))
    mask = falses(size(tokenized.mask))
    for (column, length) in enumerate(tokenized.lengths)
        mask[1:length, column] .= true
    end
    LTXTextEncoding(context, mask, copy(tokenized.lengths))
end

function _ltx_right_pad(features, mask)
    tokens, batch = size(mask)
    output = similar(features)
    output_mask = similar(mask)
    for index in 1:batch
        valid = findall(view(mask, :, index))
        padding = findall(.!view(mask, :, index))
        order = vcat(valid, padding)
        output[:, :, index] .= features[:, order, index]
        output_mask[:, index] .= mask[order, index]
    end
    output, output_mask
end

function _ltx_connector_frequencies(config::LTXTextConnectorConfig,
                                    tokens::Integer, batch::Integer, like)
    count = config.output_dim ÷ 2
    exponents = count == 1 ? Float32[0] :
        collect(range(0f0, 1f0; length=count))
    indices = config.rope_theta .^ exponents .* Float32(pi / 2)
    frequency = Array{Float32}(undef, count, tokens, batch)
    for b in 1:batch, token in 1:tokens, i in 1:count
        position = Float32(token - 1) / Float32(config.max_position)
        frequency[i, token, b] = indices[i] * (2f0 * position - 1f0)
    end
    cosine = reshape(cos.(frequency), config.head_dim ÷ 2,
                     config.heads, tokens, batch)
    sine = reshape(sin.(frequency), config.head_dim ÷ 2,
                   config.heads, tokens, batch)
    transfer(values) = begin
        source = like isa CUDA.CuArray ? CUDA.CuArray(values) : values
        cast_values(eltype(like), source)
    end
    (transfer(cosine), transfer(sine))
end

function ltx23_text_connector_forward(model::LTXTextConnector,
                                      hidden_states::AbstractVector,
                                      mask::AbstractMatrix)
    features = ltx23_gemma_features(hidden_states, mask, model.config)
    x = model.projection(features)
    x, right_mask = _ltx_right_pad(x, mask)
    if model.registers !== nothing
        size(x, 2) % size(model.registers, 2) == 0 ||
            throw(DimensionMismatch("connector token count must divide register count"))
        repeated = repeat(model.registers;
            outer=(1, size(x, 2) ÷ size(model.registers, 2)))
        for batch in 1:size(x, 3)
            invalid = findall(.!view(right_mask, :, batch))
            isempty(invalid) ||
                (x[:, invalid, batch] .= repeated[:, invalid])
        end
    end
    frequencies = _ltx_connector_frequencies(
        model.config, size(x, 2), size(x, 3), x)
    for block in model.blocks
        attended = ltx_attention_forward(block.attention,
            _ltx_rms(x, model.config.epsilon); frequencies=frequencies)
        x = mixed_add(x, attended)
        fed = block.ffn_out(gelu_tanh(
            block.ffn_in(_ltx_rms(x, model.config.epsilon))))
        x = mixed_add(x, fed)
    end
    _ltx_rms(x, model.config.epsilon)
end

function load_ltx23_text_connector(source::AbstractTensorSource,
        config::LTXTextConnectorConfig=LTXTextConnectorConfig();
        strict=false)
    specs = ltx23_text_connector_specs(config)
    audit = audit_state_dict(source, specs; allow_unexpected=!strict)
    isempty(audit.missing) && isempty(audit.shape_mismatches) ||
        _assert_clean_audit(StateDictAudit(audit.missing, String[],
                                          audit.shape_mismatches))
    model = LTXTextConnector(config; rng=Xoshiro(0), initialize=false)
    _load_dense!(source,
        "text_embedding_projection.video_aggregate_embed",
        model.projection)
    if model.registers !== nothing
        loaded = load_state_tensor(source, TensorSpec(
            "model.diffusion_model.video_embeddings_connector.learnable_registers",
            [config.registers, config.output_dim], ROW_MAJOR_SOURCE))
        # Direct BF16 -> Float32 copy can make LLVM select an unsupported
        # vector fp_extend on x86. Reuse the SafeTensors bit-exact conversion
        # path used by dense weights before copying into the FP32 model.
        _copy_model_tensor!(model.registers, permutedims(loaded))
    end
    for (index, block) in enumerate(model.blocks)
        root = "model.diffusion_model.video_embeddings_connector." *
               "transformer_1d_blocks.$(index - 1)"
        _load_ltx_attention!(source, "$root.attn1", block.attention)
        _load_dense!(source, "$root.ff.net.0.proj", block.ffn_in)
        _load_dense!(source, "$root.ff.net.2", block.ffn_out)
    end
    model
end

load_ltx23_text_connector(path::AbstractString;
                          config=ltx23_text_connector_config(
                              parse_json(inspect_safetensors(path).
                                         metadata["config"])),
                          kwargs...) =
    load_ltx23_text_connector(open_tensor_source(path), config; kwargs...)

function move_to_device(model::LTXTextConnector, transfer)
    blocks = [LTXTextConnectorBlock(
        _move_ltx_attention(block.attention, transfer),
        _move_dense(block.ffn_in, transfer),
        _move_dense(block.ffn_out, transfer)) for block in model.blocks]
    LTXTextConnector(model.config, _move_dense(model.projection, transfer),
        blocks, model.registers === nothing ? nothing :
            transfer(model.registers))
end

move_to_device(model::LTXTextConnector, device::Symbol,
               precision::Symbol=:fp32) =
    move_to_device(model, array_transfer(device, precision))
