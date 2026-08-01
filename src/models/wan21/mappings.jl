_linear_specs!(specs, prefix, output, input) = begin
    push!(specs, TensorSpec("$prefix.weight", [output, input], LINEAR_OUT_IN))
    push!(specs, TensorSpec("$prefix.bias", [output], VECTOR_LAYOUT))
end

function wan21_transformer_specs(config::Wan21Config)
    d, f = config.hidden_size, config.ffn_size
    specs = TensorSpec[]
    push!(specs, TensorSpec("patch_embedding.weight",
        [d, config.input_channels, config.patch_size...], CONV_OUT_IN_SPATIAL))
    push!(specs, TensorSpec("patch_embedding.bias", [d], VECTOR_LAYOUT))
    _linear_specs!(specs, "text_embedding.0", d, config.text_size)
    _linear_specs!(specs, "text_embedding.2", d, d)
    _linear_specs!(specs, "time_embedding.0", d, config.frequency_size)
    _linear_specs!(specs, "time_embedding.2", d, d)
    _linear_specs!(specs, "time_projection.1", 6d, d)
    for block in 0:config.layers-1
        prefix = "blocks.$block"
        push!(specs, TensorSpec("$prefix.modulation", [1, 6, d], ROW_MAJOR_SOURCE))
        for attention in ("self_attn", "cross_attn")
            for projection in ("q", "k", "v", "o")
                _linear_specs!(specs, "$prefix.$attention.$projection", d, d)
            end
            config.qk_norm && begin
                push!(specs, TensorSpec("$prefix.$attention.norm_q.weight",
                    [d], VECTOR_LAYOUT))
                push!(specs, TensorSpec("$prefix.$attention.norm_k.weight",
                    [d], VECTOR_LAYOUT))
            end
            if attention == "cross_attn" && config.model_type === :i2v
                for projection in ("k_img", "v_img")
                    _linear_specs!(
                        specs, "$prefix.$attention.$projection", d, d)
                end
                config.qk_norm && push!(specs, TensorSpec(
                    "$prefix.$attention.norm_k_img.weight",
                    [d], VECTOR_LAYOUT))
            end
        end
        if config.cross_attention_norm
            push!(specs, TensorSpec("$prefix.norm3.weight", [d], VECTOR_LAYOUT))
            push!(specs, TensorSpec("$prefix.norm3.bias", [d], VECTOR_LAYOUT))
        end
        _linear_specs!(specs, "$prefix.ffn.0", f, d)
        _linear_specs!(specs, "$prefix.ffn.2", d, f)
    end
    if config.model_type === :i2v
        push!(specs, TensorSpec(
            "img_emb.proj.0.weight", [1280], VECTOR_LAYOUT))
        push!(specs, TensorSpec(
            "img_emb.proj.0.bias", [1280], VECTOR_LAYOUT))
        _linear_specs!(specs, "img_emb.proj.1", 1280, 1280)
        _linear_specs!(specs, "img_emb.proj.3", d, 1280)
        push!(specs, TensorSpec(
            "img_emb.proj.4.weight", [d], VECTOR_LAYOUT))
        push!(specs, TensorSpec(
            "img_emb.proj.4.bias", [d], VECTOR_LAYOUT))
    end
    push!(specs, TensorSpec("head.modulation", [1, 2, d], ROW_MAJOR_SOURCE))
    _linear_specs!(specs, "head.head",
        prod(config.patch_size) * config.output_channels, d)
    specs
end
