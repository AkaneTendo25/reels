# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

const LTX23_LORA_FORMAT = "reels-ltx23-lora-v2"
const LTX23_LORA_FORMAT_V1 = "reels-ltx23-lora-v1"

function _adapt_ltx_attention(attention::LTXAttention, prefix, targets;
                              kwargs...)
    LTXAttention(
        _adapt_dense(attention.q, "$prefix.to_q", targets; kwargs...),
        _adapt_dense(attention.k, "$prefix.to_k", targets; kwargs...),
        _adapt_dense(attention.v, "$prefix.to_v", targets; kwargs...),
        _adapt_dense(attention.o, "$prefix.to_out.0", targets; kwargs...),
        attention.norm_q, attention.norm_k, attention.gate, attention.heads,
        attention.head_dim)
end

function inject_ltx23_lora(model::LTXVideoTransformer; targets,
                           rank::Integer, alpha::Real=rank, dropout::Real=0,
                           train_bias::Bool=false,
                           rank_overrides=Dict{String,Int}(),
                           alpha_overrides=Dict{String,Float32}(),
                           rng=Random.default_rng())
    rank > 0 || throw(ArgumentError("LoRA rank must be positive"))
    isfinite(alpha) && alpha > 0 ||
        throw(ArgumentError("LoRA alpha must be finite and positive"))
    all(value -> value > 0, values(rank_overrides)) ||
        throw(ArgumentError("LoRA rank overrides must be positive"))
    all(value -> isfinite(value) && value > 0, values(alpha_overrides)) ||
        throw(ArgumentError(
            "LoRA alpha overrides must be finite and positive"))
    target_list = collect(targets)
    isempty(target_list) &&
        throw(ArgumentError("at least one LTX LoRA target is required"))
    kwargs = (; rank=rank, alpha=alpha, dropout=dropout,
              train_bias=train_bias, rank_overrides=rank_overrides,
              alpha_overrides=alpha_overrides, rng=rng)
    blocks = [begin
        root = "transformer_blocks.$(index - 1)"
        LTXVideoBlock(
            _adapt_ltx_attention(block.self_attention, "$root.attn1",
                                 target_list; kwargs...),
            _adapt_ltx_attention(block.cross_attention, "$root.attn2",
                                 target_list; kwargs...),
            _adapt_dense(block.ffn_in, "$root.ff.net.0.proj",
                         target_list; kwargs...),
            _adapt_dense(block.ffn_out, "$root.ff.net.2",
                         target_list; kwargs...),
            block.scale_shift_table, block.prompt_scale_shift_table,
            block.epsilon)
    end for (index, block) in enumerate(model.blocks)]
    adapted = LTXVideoTransformer(model.config,
        _adapt_dense(model.patchify_proj, "patchify_proj",
                     target_list; kwargs...),
        model.adaln, model.prompt_adaln, blocks,
        _adapt_dense(model.output_proj, "proj_out",
                     target_list; kwargs...),
        model.output_scale_shift)
    layers = ltx23_lora_layers(adapted)
    isempty(layers) &&
        throw(ArgumentError("LoRA targets matched no LTX-2.3 linear weights"))
    _validate_lora_overrides(layers, rank_overrides, "rank")
    _validate_lora_overrides(layers, alpha_overrides, "alpha")
    adapted
end

function _collect_ltx_attention_lora!(found, prefix, attention)
    _push_lora!(found, "$prefix.to_q", attention.q)
    _push_lora!(found, "$prefix.to_k", attention.k)
    _push_lora!(found, "$prefix.to_v", attention.v)
    _push_lora!(found, "$prefix.to_out.0", attention.o)
end

function ltx23_lora_layers(model::LTXVideoTransformer)
    found = NamedTuple[]
    _push_lora!(found, "patchify_proj", model.patchify_proj)
    for (index, block) in enumerate(model.blocks)
        root = "transformer_blocks.$(index - 1)"
        _collect_ltx_attention_lora!(found, "$root.attn1",
                                     block.self_attention)
        _collect_ltx_attention_lora!(found, "$root.attn2",
                                     block.cross_attention)
        _push_lora!(found, "$root.ff.net.0.proj", block.ffn_in)
        _push_lora!(found, "$root.ff.net.2", block.ffn_out)
    end
    _push_lora!(found, "proj_out", model.output_proj)
    found
end

function ltx23_lora_parameters(model::LTXVideoTransformer)
    parameters = NamedTuple[]
    for entry in ltx23_lora_layers(model)
        push!(parameters,
            (name="diffusion_model.$(entry.path).lora_A.weight",
             value=entry.layer.A))
        push!(parameters,
            (name="diffusion_model.$(entry.path).lora_B.weight",
             value=entry.layer.B))
        entry.layer.train_bias && push!(parameters,
            (name="diffusion_model.$(entry.path).bias",
             value=entry.layer.bias))
    end
    parameters
end

function ltx23_lora_state_dict(model::LTXVideoTransformer)
    state = Dict{String,AbstractArray}()
    for entry in ltx23_lora_layers(model)
        root = "diffusion_model.$(entry.path)"
        state["$root.lora_A.weight"] = Array(entry.layer.A)
        raw_b = Array(entry.layer.B)
        scale = Float32(entry.layer.alpha / size(entry.layer.A, 1))
        state["$root.lora_B.weight"] =
            cast_values(eltype(raw_b), float32_values(raw_b) .* scale)
        entry.layer.train_bias &&
            (state["$root.bias"] = Array(entry.layer.bias))
    end
    state
end

function save_ltx23_lora(path::AbstractString, model::LTXVideoTransformer;
                         base_model="", metadata=Dict{String,String}())
    layers = ltx23_lora_layers(model)
    isempty(layers) && throw(ArgumentError("LTX model has no LoRA adapters"))
    ranks = unique(size(entry.layer.A, 1) for entry in layers)
    alphas = unique(entry.layer.alpha for entry in layers)
    info = Dict{String,String}(
        "format" => LTX23_LORA_FORMAT,
        "model_family" => "ltx23",
        "base_model" => String(base_model),
        "rank" => length(ranks) == 1 ? string(only(ranks)) : "mixed",
        "alpha" => length(alphas) == 1 ? string(only(alphas)) : "mixed",
        "train_bias" =>
            string(any(entry -> entry.layer.train_bias, layers)),
        # LTX-Core/ComfyUI fuse B@A directly and does not consume PEFT alpha.
        "scaling_baked_into_lora_B" => "true")
    merge!(info, metadata)
    write_safetensors(path, ltx23_lora_state_dict(model); metadata=info)
end

function load_ltx23_lora!(model::LTXVideoTransformer,
                          path::AbstractString; strict=true)
    source = open_tensor_source(path)
    header = source isa SingleTensorSource ? source.header :
        throw(ArgumentError("LTX LoRA import expects one SafeTensors file"))
    format = get(header.metadata, "format", "")
    format in (LTX23_LORA_FORMAT, LTX23_LORA_FORMAT_V1) ||
        throw(ArgumentError("unsupported LTX LoRA format"))
    layers = ltx23_lora_layers(model)
    specs = TensorSpec[]
    for entry in layers
        root = "diffusion_model.$(entry.path)"
        push!(specs,
            TensorSpec("$root.lora_A.weight", collect(size(entry.layer.A)),
                       LINEAR_OUT_IN),
            TensorSpec("$root.lora_B.weight", collect(size(entry.layer.B)),
                       LINEAR_OUT_IN))
        entry.layer.train_bias && push!(specs,
            TensorSpec("$root.bias", [length(entry.layer.bias)],
                       VECTOR_LAYOUT))
    end
    _assert_clean_audit(audit_state_dict(
        source, specs; allow_unexpected=!strict))
    for entry in layers
        root = "diffusion_model.$(entry.path)"
        copyto!(entry.layer.A, load_state_tensor(source,
            TensorSpec("$root.lora_A.weight", collect(size(entry.layer.A)),
                       LINEAR_OUT_IN)))
        loaded_b = load_state_tensor(source,
            TensorSpec("$root.lora_B.weight", collect(size(entry.layer.B)),
                       LINEAR_OUT_IN))
        if format == LTX23_LORA_FORMAT
            scale = Float32(entry.layer.alpha /
                            size(entry.layer.A, 1))
            scale == 0f0 &&
                throw(ArgumentError("target LTX LoRA scale is zero"))
            loaded_b = cast_values(eltype(loaded_b),
                float32_values(loaded_b) ./ scale)
        end
        copyto!(entry.layer.B, loaded_b)
        entry.layer.train_bias && copyto!(entry.layer.bias,
            load_state_tensor(source,
                TensorSpec("$root.bias", [length(entry.layer.bias)],
                           VECTOR_LAYOUT)))
    end
    model
end
