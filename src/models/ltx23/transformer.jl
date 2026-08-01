# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

struct LTXTimestepEmbedder{A,B}
    linear1::A
    linear2::B
end

struct LTXAdaLN{E,L}
    embedder::E
    linear::L
    coefficient::Int
end

struct LTXAttention{Q,K,V,O,NQ,NK,G}
    q::Q
    k::K
    v::V
    o::O
    norm_q::NQ
    norm_k::NK
    gate::G
    heads::Int
    head_dim::Int
end

struct LTXVideoBlock{SA,CA,FI,FO,M,P}
    self_attention::SA
    cross_attention::CA
    ffn_in::FI
    ffn_out::FO
    scale_shift_table::M
    prompt_scale_shift_table::P
    epsilon::Float32
end

struct LTXVideoTransformer{P,A,PA,B,O,M}
    config::LTX23Config
    patchify_proj::P
    adaln::A
    prompt_adaln::PA
    blocks::B
    output_proj::O
    output_scale_shift::M
end

function _ltx_dense(rng, output, input; initialize=true)
    weight = initialize ?
        randn(rng, Float32, output, input) ./ sqrt(Float32(input)) :
        zeros(Float32, output, input)
    DenseLayer(weight, zeros(Float32, output))
end

function _ltx_attention(config::LTX23Config, context_dim;
                        rng=Random.default_rng(), initialize=true)
    d = ltx23_video_dim(config)
    LTXAttention(
        _ltx_dense(rng, d, d; initialize=initialize),
        _ltx_dense(rng, d, context_dim; initialize=initialize),
        _ltx_dense(rng, d, context_dim; initialize=initialize),
        _ltx_dense(rng, d, d; initialize=initialize),
        RMSNorm(ones(Float32, d), config.epsilon),
        RMSNorm(ones(Float32, d), config.epsilon),
        config.gated_attention ?
            _ltx_dense(rng, config.video_heads, d; initialize=initialize) :
            nothing,
        config.video_heads, config.video_head_dim)
end

function LTXVideoTransformer(config::LTX23Config;
                             rng=Random.default_rng(), initialize=true)
    d = ltx23_video_dim(config)
    config.cross_attention_adaln && config.video_context_dim != d &&
        throw(ArgumentError("LTX cross-attention AdaLN requires context and video dimensions to match"))
    coefficient = ltx23_adaln_parameters(config)
    embedder = LTXTimestepEmbedder(
        _ltx_dense(rng, d, 256; initialize=initialize),
        _ltx_dense(rng, d, d; initialize=initialize))
    adaln = LTXAdaLN(embedder,
        _ltx_dense(rng, coefficient * d, d; initialize=initialize),
        coefficient)
    prompt_adaln = if config.cross_attention_adaln
        prompt_embedder = LTXTimestepEmbedder(
            _ltx_dense(rng, d, 256; initialize=initialize),
            _ltx_dense(rng, d, d; initialize=initialize))
        LTXAdaLN(prompt_embedder,
            _ltx_dense(rng, 2d, d; initialize=initialize), 2)
    else
        nothing
    end
    blocks = [LTXVideoBlock(
        _ltx_attention(config, d; rng=rng, initialize=initialize),
        _ltx_attention(config, config.video_context_dim;
                       rng=rng, initialize=initialize),
        _ltx_dense(rng, 4d, d; initialize=initialize),
        _ltx_dense(rng, d, 4d; initialize=initialize),
        initialize ? randn(rng, Float32, d, coefficient) ./
                     sqrt(Float32(d)) : zeros(Float32, d, coefficient),
        config.cross_attention_adaln ?
            (initialize ? randn(rng, Float32, d, 2) /
                          sqrt(Float32(d)) : zeros(Float32, d, 2)) :
            nothing,
        config.epsilon) for _ in 1:config.layers]
    LTXVideoTransformer(config,
        _ltx_dense(rng, d, config.video_channels; initialize=initialize),
        adaln, prompt_adaln, blocks,
        _ltx_dense(rng, config.video_channels, d; initialize=initialize),
        initialize ? randn(rng, Float32, d, 2) ./ sqrt(Float32(d)) :
                     zeros(Float32, d, 2))
end

function ltx_timestep_embedding(timesteps::AbstractVector,
                                dimension::Integer=256)
    iseven(dimension) ||
        throw(ArgumentError("LTX timestep embedding dimension must be even"))
    half = dimension ÷ 2
    result = Matrix{Float32}(undef, dimension, length(timesteps))
    for batch in eachindex(timesteps), index in 0:half-1
        frequency = exp(-log(10_000f0) * Float32(index) / Float32(half))
        angle = Float32(timesteps[batch]) * frequency
        # Official Timesteps uses flip_sin_to_cos=true.
        result[index + 1, batch] = cos(angle)
        result[half + index + 1, batch] = sin(angle)
    end
    result
end

function ltx_adaln_forward(adaln::LTXAdaLN, timesteps::AbstractVector,
                           like)
    host = ltx_timestep_embedding(Array(timesteps))
    projected = _constant_like(like, host)
    embedded = adaln.embedder.linear2(
        _ltx_silu(adaln.embedder.linear1(projected)))
    modulation = adaln.linear(_ltx_silu(embedded))
    (reshape(modulation, size(embedded, 1), adaln.coefficient,
             size(embedded, 2)), embedded)
end

function _ltx_scaled_timesteps(timesteps, scale::Real, like)
    # CUDA's mixed Float32 .* BFloat16 broadcast can silently compile to an
    # invalid fp-extend kernel. Widen the values explicitly before scaling.
    scaled32 = Float32(scale) .* float32_values(timesteps)
    cast_values(eltype(like), scaled32)
end

function _ltx_silu(x)
    x32 = float32_values(x)
    y = x32 ./ (1f0 .+ exp.(-x32))
    eltype(x) === BFloat16 ? bfloat16_values(y) :
        cast_values(eltype(x), y)
end

function _ltx_rms(x, epsilon)
    x32 = float32_values(x)
    normalized = x32 ./ sqrt.(sum(abs2, x32; dims=1) ./
                              Float32(size(x, 1)) .+ Float32(epsilon))
    cast_values(eltype(x), normalized)
end

"""
Generate official split-RoPE frequencies from midpoint positions.
`positions` has layout `(axes, tokens, batch)` and axes `(time,height,width)`.
"""
function ltx_rope_frequencies(config::LTX23Config,
                              positions::AbstractArray{<:Real,3}, like)
    size(positions, 1) == 3 ||
        throw(DimensionMismatch("LTX video positions require three axes"))
    d = ltx23_video_dim(config)
    count = d ÷ 6
    count > 0 || throw(ArgumentError("LTX hidden dimension is too small for 3D RoPE"))
    exponents = count == 1 ? Float32[0] :
        collect(range(0f0, 1f0; length=count))
    indices = config.rope_theta .^ exponents .* Float32(pi / 2)
    tokens, batch = size(positions, 2), size(positions, 3)
    host_positions = positions isa CUDA.CuArray ?
        Array(positions) : positions
    frequencies = Array{Float32}(undef, 3count, tokens, batch)
    maxima = config.video_max_positions
    for b in 1:batch, token in 1:tokens, axis in 1:3, i in 1:count
        fractional = Float32(host_positions[axis, token, b]) /
                     Float32(maxima[axis])
        # Upstream flattens `(frequency, axis)`, so each frequency's
        # time/height/width triplet is contiguous.
        frequencies[(i - 1) * 3 + axis, token, b] =
            indices[i] * (2f0 * fractional - 1f0)
    end
    needed = d ÷ 2
    pad = needed - size(frequencies, 1)
    pad >= 0 || throw(ArgumentError("invalid LTX RoPE frequency allocation"))
    padded = vcat(zeros(Float32, pad, tokens, batch), frequencies)
    cos_host = reshape(cos.(padded), config.video_head_dim ÷ 2,
                       config.video_heads, tokens, batch)
    sin_host = reshape(sin.(padded), config.video_head_dim ÷ 2,
                       config.video_heads, tokens, batch)
    transfer(values) = begin
        source = like isa CUDA.CuArray ? CUDA.CuArray(values) : values
        cast_values(eltype(like), source)
    end
    (transfer(cos_host), transfer(sin_host))
end

function ltx_apply_rope(x::AbstractArray{T,4}, frequencies) where T
    iseven(size(x, 1)) ||
        throw(DimensionMismatch("LTX RoPE head dimension must be even"))
    cosine, sine = frequencies
    half = size(x, 1) ÷ 2
    first = view(x, 1:half, :, :, :)
    second = view(x, half+1:size(x, 1), :, :, :)
    rotated_first = mixed_sub(mixed_mul(first, cosine),
                              mixed_mul(second, sine))
    rotated_second = mixed_add(mixed_mul(second, cosine),
                               mixed_mul(first, sine))
    vcat(rotated_first, rotated_second)
end

function ltx_attention_forward(attention::LTXAttention, x,
                               context=x; frequencies=nothing,
                               mask=nothing, training::Bool=false,
                               dropout_seed::UInt64=UInt64(0),
                               dropout_stream::Integer=0)
    size(x, 1) == size(attention.q.weight, 2) ||
        throw(DimensionMismatch("LTX query feature dimension differs"))
    q = rmsnorm(attention.norm_q, _projection(attention.q, x;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 1))
    k = rmsnorm(attention.norm_k, _projection(attention.k, context;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 2))
    v = _projection(attention.v, context;
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 3)
    q = reshape(q, attention.head_dim, attention.heads,
                size(q, 2), size(q, 3))
    k = reshape(k, attention.head_dim, attention.heads,
                size(k, 2), size(k, 3))
    v = reshape(v, attention.head_dim, attention.heads,
                size(v, 2), size(v, 3))
    if frequencies !== nothing
        q = ltx_apply_rope(q, frequencies)
        k = ltx_apply_rope(k, frequencies)
    end
    attended = memory_efficient_attention(q, k, v; mask=mask).output
    if attention.gate !== nothing
        gate_logits = _projection(attention.gate, x;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=dropout_stream + 4)
        gate32 = float32_values(gate_logits)
        gates = cast_values(eltype(gate_logits),
            2f0 ./ (1f0 .+ exp.(-gate32)))
        attended = mixed_mul(attended,
            reshape(gates, 1, attention.heads,
                    size(gates, 2), size(gates, 3)))
    end
    _projection(attention.o,
        reshape(attended, attention.head_dim * attention.heads,
            size(attended, 3), size(attended, 4));
        training=training, dropout_seed=dropout_seed,
        dropout_stream=dropout_stream + 5)
end

function _ltx_modulation(table, timestep, index)
    value = mixed_add(view(timestep, :, index, :),
                      reshape(view(table, :, index), :, 1))
    reshape(value, size(value, 1), 1, size(value, 2))
end

function ltx_video_block_forward(block::LTXVideoBlock, x, timestep,
                                 context, frequencies,
                                 prompt_timestep=nothing,
                                 training::Bool=false,
                                 dropout_seed::UInt64=UInt64(0))
    shift_sa = _ltx_modulation(block.scale_shift_table, timestep, 1)
    scale_sa = _ltx_modulation(block.scale_shift_table, timestep, 2)
    gate_sa = _ltx_modulation(block.scale_shift_table, timestep, 3)
    normalized = mixed_affine(_ltx_rms(x, block.epsilon),
                              scale_sa, shift_sa)
    self_output = ltx_attention_forward(block.self_attention, normalized;
        frequencies=frequencies, training=training,
        dropout_seed=dropout_seed, dropout_stream=10)
    x = mixed_add(x, mixed_mul(self_output, gate_sa))
    normalized_cross = _ltx_rms(x, block.epsilon)
    cross_output = if prompt_timestep === nothing
        ltx_attention_forward(
            block.cross_attention, normalized_cross, context;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=20)
    else
        shift_q = _ltx_modulation(block.scale_shift_table, timestep, 7)
        scale_q = _ltx_modulation(block.scale_shift_table, timestep, 8)
        gate_q = _ltx_modulation(block.scale_shift_table, timestep, 9)
        shift_kv = _ltx_modulation(block.prompt_scale_shift_table,
                                   prompt_timestep, 1)
        scale_kv = _ltx_modulation(block.prompt_scale_shift_table,
                                   prompt_timestep, 2)
        query = mixed_affine(normalized_cross, scale_q, shift_q)
        keys = mixed_affine(context, scale_kv, shift_kv)
        mixed_mul(ltx_attention_forward(
            block.cross_attention, query, keys;
            training=training, dropout_seed=dropout_seed,
            dropout_stream=20), gate_q)
    end
    x = mixed_add(x, cross_output)

    shift_ff = _ltx_modulation(block.scale_shift_table, timestep, 4)
    scale_ff = _ltx_modulation(block.scale_shift_table, timestep, 5)
    gate_ff = _ltx_modulation(block.scale_shift_table, timestep, 6)
    ffn_input = mixed_affine(_ltx_rms(x, block.epsilon),
                             scale_ff, shift_ff)
    hidden = _projection(block.ffn_in, ffn_input;
        training=training, dropout_seed=dropout_seed, dropout_stream=30)
    ffn_output = _projection(block.ffn_out, gelu_tanh(hidden);
        training=training, dropout_seed=dropout_seed, dropout_stream=31)
    mixed_add(x, mixed_mul(ffn_output, gate_ff))
end

function ltx_video_transformer_forward(model::LTXVideoTransformer,
                                       latents::AbstractArray{T,3},
                                       timesteps::AbstractVector,
                                       context::AbstractArray{S,3},
                                       positions::AbstractArray{<:Real,3};
                                       checkpoint_interval::Int=0,
                                       sigma=timesteps,
                                       training::Bool=false,
                                       dropout_seed::UInt64=UInt64(0)) where {T,S}
    checkpoint_interval >= 0 ||
        throw(ArgumentError("checkpoint_interval cannot be negative"))
    size(latents, 3) == length(timesteps) == size(context, 3) ==
        size(positions, 3) || throw(DimensionMismatch("LTX batch dimensions differ"))
    size(latents, 2) == size(positions, 2) ||
        throw(DimensionMismatch("LTX token and position counts differ"))
    size(context, 1) == model.config.video_context_dim ||
        throw(DimensionMismatch("LTX context feature dimension differs"))
    x = _projection(model.patchify_proj, latents;
        training=training, dropout_seed=dropout_seed, dropout_stream=1)
    scaled_timesteps = _ltx_scaled_timesteps(
        timesteps, model.config.timestep_scale, x)
    timestep, embedded = ltx_adaln_forward(
        model.adaln, scaled_timesteps, x)
    prompt_timestep = model.prompt_adaln === nothing ? nothing :
        first(ltx_adaln_forward(model.prompt_adaln,
            _ltx_scaled_timesteps(sigma, model.config.timestep_scale, x), x))
    frequencies = ltx_rope_frequencies(model.config, positions, x)
    for (index, block) in enumerate(model.blocks)
        block_seed = _dropout_seed(dropout_seed, UInt64(100 + index))
        if checkpoint_interval > 0 &&
           (index - 1) % checkpoint_interval == 0
            x = Zygote.checkpointed(ltx_video_block_forward, block, x,
                                    timestep, context, frequencies,
                                    prompt_timestep, training, block_seed)
        else
            x = ltx_video_block_forward(block, x, timestep,
                                        context, frequencies,
                                        prompt_timestep, training, block_seed)
        end
    end
    shift = reshape(mixed_add(embedded,
        reshape(model.output_scale_shift[:, 1], :, 1)), :, 1, size(x, 3))
    scale = reshape(mixed_add(embedded,
        reshape(model.output_scale_shift[:, 2], :, 1)), :, 1, size(x, 3))
    _projection(model.output_proj, mixed_affine(
        layernorm(x; epsilon=model.config.epsilon), scale, shift);
        training=training, dropout_seed=dropout_seed, dropout_stream=1000)
end

function _move_ltx_attention(attention, transfer)
    LTXAttention(_move_dense(attention.q, transfer),
        _move_dense(attention.k, transfer), _move_dense(attention.v, transfer),
        _move_dense(attention.o, transfer),
        RMSNorm(transfer(attention.norm_q.weight), attention.norm_q.epsilon),
        RMSNorm(transfer(attention.norm_k.weight), attention.norm_k.epsilon),
        attention.gate === nothing ? nothing :
            _move_dense(attention.gate, transfer),
        attention.heads, attention.head_dim)
end

function move_to_device(model::LTXVideoTransformer, transfer)
    embedder = LTXTimestepEmbedder(
        _move_dense(model.adaln.embedder.linear1, transfer),
        _move_dense(model.adaln.embedder.linear2, transfer))
    adaln = LTXAdaLN(embedder, _move_dense(model.adaln.linear, transfer),
                     model.adaln.coefficient)
    prompt_adaln = model.prompt_adaln === nothing ? nothing : begin
        prompt_embedder = LTXTimestepEmbedder(
            _move_dense(model.prompt_adaln.embedder.linear1, transfer),
            _move_dense(model.prompt_adaln.embedder.linear2, transfer))
        LTXAdaLN(prompt_embedder,
            _move_dense(model.prompt_adaln.linear, transfer),
            model.prompt_adaln.coefficient)
    end
    blocks = [LTXVideoBlock(
        _move_ltx_attention(block.self_attention, transfer),
        _move_ltx_attention(block.cross_attention, transfer),
        _move_dense(block.ffn_in, transfer),
        _move_dense(block.ffn_out, transfer),
        transfer(block.scale_shift_table),
        block.prompt_scale_shift_table === nothing ? nothing :
            transfer(block.prompt_scale_shift_table),
        block.epsilon)
        for block in model.blocks]
    LTXVideoTransformer(model.config,
        _move_dense(model.patchify_proj, transfer), adaln, prompt_adaln, blocks,
        _move_dense(model.output_proj, transfer),
        transfer(model.output_scale_shift))
end

move_to_device(model::LTXVideoTransformer, device::Symbol,
               precision::Symbol=:fp32) =
    move_to_device(model, array_transfer(device, precision))

function _load_ltx_attention!(source, prefix, attention)
    _load_dense!(source, "$prefix.to_q", attention.q)
    _load_dense!(source, "$prefix.to_k", attention.k)
    _load_dense!(source, "$prefix.to_v", attention.v)
    _load_dense!(source, "$prefix.to_out.0", attention.o)
    _copy_model_tensor!(attention.norm_q.weight, load_state_tensor(source,
        TensorSpec("$prefix.q_norm.weight",
                   collect(size(attention.norm_q.weight)), VECTOR_LAYOUT)))
    _copy_model_tensor!(attention.norm_k.weight, load_state_tensor(source,
        TensorSpec("$prefix.k_norm.weight",
                   collect(size(attention.norm_k.weight)), VECTOR_LAYOUT)))
    attention.gate === nothing ||
        _load_dense!(source, "$prefix.to_gate_logits", attention.gate)
end

function _load_ltx_adaln!(source, prefix, adaln)
    _load_dense!(source, "$prefix.emb.timestep_embedder.linear_1",
                 adaln.embedder.linear1)
    _load_dense!(source, "$prefix.emb.timestep_embedder.linear_2",
                 adaln.embedder.linear2)
    _load_dense!(source, "$prefix.linear", adaln.linear)
end

function load_ltx23_video_transformer(source::AbstractTensorSource,
                                      config::LTX23Config;
                                      strict=true,
                                      checkpoint_prefix="")
    video_config = config.video_only ? config :
        LTX23Config(variant=config.variant, video_only=true,
            video_heads=config.video_heads, video_head_dim=config.video_head_dim,
            video_channels=config.video_channels,
            video_context_dim=config.video_context_dim, audio_heads=config.audio_heads,
            audio_head_dim=config.audio_head_dim, audio_channels=config.audio_channels,
            audio_context_dim=config.audio_context_dim, layers=config.layers,
            timestep_scale=config.timestep_scale, rope_theta=config.rope_theta,
            video_max_positions=config.video_max_positions,
            audio_max_positions=config.audio_max_positions,
            cross_attention_adaln=config.cross_attention_adaln,
            gated_attention=config.gated_attention, epsilon=config.epsilon)
    specs = ltx23_transformer_specs(video_config;
                                    checkpoint_prefix=checkpoint_prefix)
    audit = audit_state_dict(source, specs; allow_unexpected=!strict)
    _assert_clean_audit(audit)
    model = LTXVideoTransformer(video_config; initialize=false)
    prefix = String(checkpoint_prefix)
    _load_dense!(source, prefix * "patchify_proj", model.patchify_proj)
    _load_ltx_adaln!(source, prefix * "adaln_single", model.adaln)
    model.prompt_adaln === nothing ||
        _load_ltx_adaln!(source, prefix * "prompt_adaln_single",
                         model.prompt_adaln)
    for (index, block) in enumerate(model.blocks)
        root = prefix * "transformer_blocks.$(index - 1)"
        _load_ltx_attention!(source, "$root.attn1", block.self_attention)
        _load_ltx_attention!(source, "$root.attn2", block.cross_attention)
        _load_dense!(source, "$root.ff.net.0.proj", block.ffn_in)
        _load_dense!(source, "$root.ff.net.2", block.ffn_out)
        table = load_state_tensor(source,
            TensorSpec("$root.scale_shift_table",
                [model.adaln.coefficient, ltx23_video_dim(video_config)],
                ROW_MAJOR_SOURCE))
        _copy_model_tensor!(block.scale_shift_table, permutedims(table))
        if block.prompt_scale_shift_table !== nothing
            prompt_table = load_state_tensor(source,
                TensorSpec("$root.prompt_scale_shift_table",
                    [2, ltx23_video_dim(video_config)], ROW_MAJOR_SOURCE))
            _copy_model_tensor!(
                block.prompt_scale_shift_table, permutedims(prompt_table))
        end
    end
    table = load_state_tensor(source, TensorSpec(prefix * "scale_shift_table",
        [2, ltx23_video_dim(video_config)], ROW_MAJOR_SOURCE))
    _copy_model_tensor!(model.output_scale_shift, permutedims(table))
    _load_dense!(source, prefix * "proj_out", model.output_proj)
    model
end

load_ltx23_video_transformer(path::AbstractString,
                             config::LTX23Config; kwargs...) =
    load_ltx23_video_transformer(open_tensor_source(path), config; kwargs...)
