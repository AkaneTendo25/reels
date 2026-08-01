const WAN_LORA_FORMAT = "reels-wan21-lora-v2"
const WAN_LORA_FORMAT_V1 = "reels-wan21-lora-v1"

_target_matches(target::Regex, key::String) = occursin(target, key)
_target_matches(target::AbstractString, key::String) = target == key
_target_matches(target, key::String) =
    throw(ArgumentError("LoRA target must be a Regex or string, got $(typeof(target))"))

function _adapt_dense(layer::DenseLayer, path::String, targets;
                      rank, alpha, dropout, train_bias=false,
                      rank_overrides=Dict{String,Int}(),
                      alpha_overrides=Dict{String,Float32}(), rng)
    key = "$path.weight"
    any(target -> _target_matches(target, key), targets) || return layer
    module_rank = get(rank_overrides, key, get(rank_overrides, path, rank))
    module_alpha = get(
        alpha_overrides, key, get(alpha_overrides, path, alpha))
    LoRALinear(layer.weight; rank=module_rank, alpha=module_alpha,
               dropout=dropout, bias=layer.bias, train_bias=train_bias,
               rng=rng)
end
_adapt_dense(layer::LoRALinear, path::String, targets; kwargs...) =
    throw(ArgumentError("projection $path already has a LoRA adapter"))

function _adapt_attention(attention::WanAttention, prefix, targets; kwargs...)
    WanAttention(
        _adapt_dense(attention.q, "$prefix.q", targets; kwargs...),
        _adapt_dense(attention.k, "$prefix.k", targets; kwargs...),
        _adapt_dense(attention.v, "$prefix.v", targets; kwargs...),
        _adapt_dense(attention.o, "$prefix.o", targets; kwargs...),
        attention.norm_q, attention.norm_k,
        attention.k_img === nothing ? nothing :
            _adapt_dense(
                attention.k_img, "$prefix.k_img", targets; kwargs...),
        attention.v_img === nothing ? nothing :
            _adapt_dense(
                attention.v_img, "$prefix.v_img", targets; kwargs...),
        attention.norm_k_img, attention.heads)
end

function _adapt_image_projection(projection::WanImageProjection,
                                 targets; kwargs...)
    WanImageProjection(
        projection.input_norm_weight, projection.input_norm_bias,
        _adapt_dense(projection.input_projection, "img_emb.proj.1",
                     targets; kwargs...),
        _adapt_dense(projection.output_projection, "img_emb.proj.3",
                     targets; kwargs...),
        projection.output_norm_weight, projection.output_norm_bias,
        projection.epsilon)
end

function _adapt_block(block::WanTransformerBlock, prefix, targets; kwargs...)
    WanTransformerBlock(
        _adapt_attention(block.self_attention, "$prefix.self_attn", targets;
                         kwargs...),
        _adapt_attention(block.cross_attention, "$prefix.cross_attn", targets;
                         kwargs...),
        _adapt_dense(block.ffn_in, "$prefix.ffn.0", targets; kwargs...),
        _adapt_dense(block.ffn_out, "$prefix.ffn.2", targets; kwargs...),
        block.modulation, block.cross_norm_weight, block.cross_norm_bias,
        block.epsilon)
end

"""
    inject_wan_lora(model; targets, rank, alpha=rank, dropout=0, rng)

Return a Wan transformer sharing its frozen base arrays with `model` and with
fresh LoRA adapters inserted into every matching official state-dict key.
Targets are exact strings or regular expressions and are matched against keys
ending in `.weight`.
"""
function _validate_lora_overrides(layers, overrides, label)
    available = Set{String}()
    for entry in layers
        push!(available, entry.path)
        push!(available, "$(entry.path).weight")
    end
    unknown = sort!(setdiff(String.(collect(keys(overrides))), available))
    isempty(unknown) ||
        throw(ArgumentError(
            "$label overrides matched no injected LoRA module: " *
            join(unknown, ", ")))
    nothing
end

function inject_wan_lora(model::WanTransformer; targets, rank::Integer,
                         alpha::Real=rank, dropout::Real=0,
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
    isempty(target_list) && throw(ArgumentError("at least one LoRA target is required"))
    blocks = [_adapt_block(block, "blocks.$(index - 1)", target_list;
        rank=rank, alpha=alpha, dropout=dropout, train_bias=train_bias,
        rank_overrides=rank_overrides, alpha_overrides=alpha_overrides,
        rng=rng)
        for (index, block) in enumerate(model.blocks)]
    adapted = WanTransformer(model.config, model.patch_weight, model.patch_bias,
        _adapt_dense(model.text_in, "text_embedding.0", target_list;
                     rank=rank, alpha=alpha, dropout=dropout,
                     train_bias=train_bias, rank_overrides=rank_overrides,
                     alpha_overrides=alpha_overrides, rng=rng),
        _adapt_dense(model.text_out, "text_embedding.2", target_list;
                     rank=rank, alpha=alpha, dropout=dropout,
                     train_bias=train_bias, rank_overrides=rank_overrides,
                     alpha_overrides=alpha_overrides, rng=rng),
        _adapt_dense(model.time_in, "time_embedding.0", target_list;
                     rank=rank, alpha=alpha, dropout=dropout,
                     train_bias=train_bias, rank_overrides=rank_overrides,
                     alpha_overrides=alpha_overrides, rng=rng),
        _adapt_dense(model.time_out, "time_embedding.2", target_list;
                     rank=rank, alpha=alpha, dropout=dropout,
                     train_bias=train_bias, rank_overrides=rank_overrides,
                     alpha_overrides=alpha_overrides, rng=rng),
        _adapt_dense(model.time_projection, "time_projection.1", target_list;
                     rank=rank, alpha=alpha, dropout=dropout,
                     train_bias=train_bias, rank_overrides=rank_overrides,
                     alpha_overrides=alpha_overrides, rng=rng),
        blocks,
        _adapt_dense(model.head, "head.head", target_list;
                     rank=rank, alpha=alpha, dropout=dropout,
                     train_bias=train_bias, rank_overrides=rank_overrides,
                     alpha_overrides=alpha_overrides, rng=rng),
        model.head_modulation,
        model.image_projection === nothing ? nothing :
            _adapt_image_projection(
                model.image_projection, target_list;
                rank=rank, alpha=alpha, dropout=dropout,
                train_bias=train_bias, rank_overrides=rank_overrides,
                alpha_overrides=alpha_overrides, rng=rng))
    layers = wan_lora_layers(adapted)
    isempty(layers) &&
        throw(ArgumentError("LoRA targets matched no Wan linear weights"))
    _validate_lora_overrides(layers, rank_overrides, "rank")
    _validate_lora_overrides(layers, alpha_overrides, "alpha")
    adapted
end

function _push_lora!(found, path, layer)
    layer isa LoRALinear && push!(found, (path=String(path), layer=layer))
end
function _collect_attention_lora!(found, prefix, attention)
    _push_lora!(found, "$prefix.q", attention.q)
    _push_lora!(found, "$prefix.k", attention.k)
    _push_lora!(found, "$prefix.v", attention.v)
    _push_lora!(found, "$prefix.o", attention.o)
    _push_lora!(found, "$prefix.k_img", attention.k_img)
    _push_lora!(found, "$prefix.v_img", attention.v_img)
end

"""Return LoRA layers in deterministic official state-dict path order."""
function wan_lora_layers(model::WanTransformer)
    found = NamedTuple[]
    _push_lora!(found, "text_embedding.0", model.text_in)
    _push_lora!(found, "text_embedding.2", model.text_out)
    _push_lora!(found, "time_embedding.0", model.time_in)
    _push_lora!(found, "time_embedding.2", model.time_out)
    _push_lora!(found, "time_projection.1", model.time_projection)
    for (index, block) in enumerate(model.blocks)
        prefix = "blocks.$(index - 1)"
        _collect_attention_lora!(found, "$prefix.self_attn", block.self_attention)
        _collect_attention_lora!(found, "$prefix.cross_attn", block.cross_attention)
        _push_lora!(found, "$prefix.ffn.0", block.ffn_in)
        _push_lora!(found, "$prefix.ffn.2", block.ffn_out)
    end
    _push_lora!(found, "head.head", model.head)
    if model.image_projection !== nothing
        _push_lora!(
            found, "img_emb.proj.1",
            model.image_projection.input_projection)
        _push_lora!(
            found, "img_emb.proj.3",
            model.image_projection.output_projection)
    end
    found
end

"""Flat, stable `(name, array)` entries suitable for optimizers."""
function wan_lora_parameters(model::WanTransformer)
    parameters = NamedTuple[]
    for entry in wan_lora_layers(model)
        push!(parameters, (name="$(entry.path).lora_A.weight", value=entry.layer.A))
        push!(parameters, (name="$(entry.path).lora_B.weight", value=entry.layer.B))
        entry.layer.train_bias &&
            push!(parameters,
                (name="$(entry.path).bias", value=entry.layer.bias))
    end
    parameters
end

function wan_lora_state_dict(model::WanTransformer)
    state = Dict{String,AbstractArray}()
    for entry in wan_lora_layers(model)
        prefix = "diffusion_model.$(entry.path)"
        state["$prefix.lora_A.weight"] = Array(entry.layer.A)
        raw_b = Array(entry.layer.B)
        scale = Float32(entry.layer.alpha / size(entry.layer.A, 1))
        state["$prefix.lora_B.weight"] =
            cast_values(eltype(raw_b), float32_values(raw_b) .* scale)
        entry.layer.train_bias &&
            (state["$prefix.bias"] = Array(entry.layer.bias))
    end
    state
end

function save_wan_lora(path::AbstractString, model::WanTransformer;
                       base_model="", metadata=Dict{String,String}())
    layers = wan_lora_layers(model)
    isempty(layers) && throw(ArgumentError("Wan model has no LoRA adapters"))
    ranks = unique(size(entry.layer.A, 1) for entry in layers)
    alphas = unique(entry.layer.alpha for entry in layers)
    info = Dict{String,String}(
        "format" => WAN_LORA_FORMAT,
        "model_family" => "wan21",
        "base_model" => String(base_model),
        "rank" => length(ranks) == 1 ? string(only(ranks)) : "mixed",
        "alpha" => length(alphas) == 1 ? string(only(alphas)) : "mixed",
        "scaling_baked_into_lora_B" => "true",
        "external_format" => "diffusers-non-diffusers-wan",
        "train_bias" =>
            string(any(entry -> entry.layer.train_bias, layers)),
    )
    merge!(info, metadata)
    write_safetensors(path, wan_lora_state_dict(model); metadata=info)
end

function load_wan_lora!(model::WanTransformer, path::AbstractString; strict=true)
    source = open_tensor_source(path)
    header = source isa SingleTensorSource ? source.header :
        throw(ArgumentError("Wan LoRA import currently expects one SafeTensors file"))
    format = get(header.metadata, "format", "")
    format in (WAN_LORA_FORMAT, WAN_LORA_FORMAT_V1) ||
        throw(ArgumentError("unsupported Wan LoRA format"))
    prefix = format == WAN_LORA_FORMAT ? "diffusion_model." : ""
    layers = wan_lora_layers(model)
    isempty(layers) && throw(ArgumentError("target Wan model has no LoRA adapters"))
    specs = TensorSpec[]
    for entry in layers
        push!(specs,
            TensorSpec("$prefix$(entry.path).lora_A.weight", collect(size(entry.layer.A)),
                       LINEAR_OUT_IN),
            TensorSpec("$prefix$(entry.path).lora_B.weight", collect(size(entry.layer.B)),
                       LINEAR_OUT_IN))
        entry.layer.train_bias && push!(specs,
            TensorSpec("$prefix$(entry.path).bias",
                [length(entry.layer.bias)], VECTOR_LAYOUT))
    end
    audit = audit_state_dict(source, specs; allow_unexpected=!strict)
    _assert_clean_audit(audit)
    for entry in layers
        copyto!(entry.layer.A, load_state_tensor(source,
            TensorSpec("$prefix$(entry.path).lora_A.weight",
                       collect(size(entry.layer.A)), LINEAR_OUT_IN)))
        loaded_b = load_state_tensor(source,
            TensorSpec("$prefix$(entry.path).lora_B.weight",
                       collect(size(entry.layer.B)), LINEAR_OUT_IN))
        if format == WAN_LORA_FORMAT
            scale = Float32(entry.layer.alpha /
                            size(entry.layer.A, 1))
            scale == 0f0 &&
                throw(ArgumentError("target Wan LoRA scale is zero"))
            loaded_b = cast_values(eltype(loaded_b),
                float32_values(loaded_b) ./ scale)
        end
        copyto!(entry.layer.B, loaded_b)
        entry.layer.train_bias && copyto!(entry.layer.bias,
            load_state_tensor(source,
                TensorSpec("$prefix$(entry.path).bias",
                    [length(entry.layer.bias)], VECTOR_LAYOUT)))
    end
    model
end

# Base-checkpoint serialization intentionally omits adapter tensors.
function _store_dense!(state, prefix, layer::LoRALinear)
    state["$prefix.weight"] = copy(layer.weight)
    base_bias = layer.train_bias ? layer.base_bias : layer.bias
    base_bias === nothing || (state["$prefix.bias"] = copy(base_bias))
end

_move_dense(layer::LoRALinear, transfer) = LoRALinear(
    _move_weight(layer.weight, transfer), _move_array(layer.bias, transfer),
    transfer(layer.A), transfer(layer.B), layer.alpha, layer.dropout,
    layer.enabled, layer.train_bias, _move_array(layer.base_bias, transfer))
