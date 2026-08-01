# Independent Julia implementation of LTX-2.3 checkpoint interoperability.

function _ltx_linear_specs!(specs, prefix, output, input)
    push!(specs, TensorSpec("$prefix.weight", [output, input], LINEAR_OUT_IN))
    push!(specs, TensorSpec("$prefix.bias", [output], VECTOR_LAYOUT))
end

function _ltx_attention_specs!(specs, prefix, query_dim, context_dim,
                               inner_dim; gated=false, heads=1)
    _ltx_linear_specs!(specs, "$prefix.to_q", inner_dim, query_dim)
    _ltx_linear_specs!(specs, "$prefix.to_k", inner_dim, context_dim)
    _ltx_linear_specs!(specs, "$prefix.to_v", inner_dim, context_dim)
    push!(specs, TensorSpec("$prefix.q_norm.weight", [inner_dim], VECTOR_LAYOUT))
    push!(specs, TensorSpec("$prefix.k_norm.weight", [inner_dim], VECTOR_LAYOUT))
    gated && _ltx_linear_specs!(specs, "$prefix.to_gate_logits",
                                heads, query_dim)
    _ltx_linear_specs!(specs, "$prefix.to_out.0", query_dim, inner_dim)
end

function _ltx_ff_specs!(specs, prefix, dim)
    _ltx_linear_specs!(specs, "$prefix.net.0.proj", 4dim, dim)
    _ltx_linear_specs!(specs, "$prefix.net.2", dim, 4dim)
end

function _ltx_adaln_specs!(specs, prefix, dim, coefficient)
    _ltx_linear_specs!(specs,
        "$prefix.emb.timestep_embedder.linear_1", dim, 256)
    _ltx_linear_specs!(specs,
        "$prefix.emb.timestep_embedder.linear_2", dim, dim)
    _ltx_linear_specs!(specs, "$prefix.linear", coefficient * dim, dim)
end

"""
Return the exact official LTX-Core transformer inventory. `checkpoint_prefix`
supports monolithic checkpoints whose keys begin with `model.diffusion_model.`.
"""
function ltx23_transformer_specs(config::LTX23Config;
                                 checkpoint_prefix::AbstractString="")
    specs = TensorSpec[]
    d = ltx23_video_dim(config)
    a = ltx23_audio_dim(config)
    n_ada = ltx23_adaln_parameters(config)
    key(name) = String(checkpoint_prefix) * name

    _ltx_linear_specs!(specs, key("patchify_proj"), d, config.video_channels)
    _ltx_adaln_specs!(specs, key("adaln_single"), d, n_ada)
    config.cross_attention_adaln && begin
        _ltx_adaln_specs!(specs, key("prompt_adaln_single"), d, 2)
    end

    if !config.video_only
        _ltx_linear_specs!(specs, key("audio_patchify_proj"), a,
                           config.audio_channels)
        _ltx_adaln_specs!(specs, key("audio_adaln_single"), a, n_ada)
        _ltx_adaln_specs!(specs,
            key("av_ca_video_scale_shift_adaln_single"), d, 4)
        _ltx_adaln_specs!(specs,
            key("av_ca_audio_scale_shift_adaln_single"), a, 4)
        _ltx_adaln_specs!(specs, key("av_ca_a2v_gate_adaln_single"), d, 1)
        _ltx_adaln_specs!(specs, key("av_ca_v2a_gate_adaln_single"), a, 1)
        config.cross_attention_adaln &&
            _ltx_adaln_specs!(specs, key("audio_prompt_adaln_single"), a, 2)
    end

    for block in 0:config.layers-1
        root = key("transformer_blocks.$block")
        _ltx_attention_specs!(specs, "$root.attn1", d, d, d;
                              gated=config.gated_attention,
                              heads=config.video_heads)
        _ltx_attention_specs!(specs, "$root.attn2", d,
                              config.video_context_dim, d;
                              gated=config.gated_attention,
                              heads=config.video_heads)
        _ltx_ff_specs!(specs, "$root.ff", d)
        push!(specs, TensorSpec("$root.scale_shift_table", [n_ada, d],
                                ROW_MAJOR_SOURCE))
        config.cross_attention_adaln &&
            push!(specs, TensorSpec("$root.prompt_scale_shift_table", [2, d],
                                    ROW_MAJOR_SOURCE))
        if !config.video_only
            _ltx_attention_specs!(specs, "$root.audio_attn1", a, a, a;
                                  gated=config.gated_attention,
                                  heads=config.audio_heads)
            _ltx_attention_specs!(specs, "$root.audio_attn2", a,
                                  config.audio_context_dim, a;
                                  gated=config.gated_attention,
                                  heads=config.audio_heads)
            _ltx_ff_specs!(specs, "$root.audio_ff", a)
            push!(specs, TensorSpec("$root.audio_scale_shift_table",
                                    [n_ada, a], ROW_MAJOR_SOURCE))
            _ltx_attention_specs!(specs, "$root.audio_to_video_attn",
                                  d, a, a; gated=config.gated_attention,
                                  heads=config.audio_heads)
            _ltx_attention_specs!(specs, "$root.video_to_audio_attn",
                                  a, d, a; gated=config.gated_attention,
                                  heads=config.audio_heads)
            push!(specs, TensorSpec("$root.scale_shift_table_a2v_ca_audio",
                                    [5, a], ROW_MAJOR_SOURCE))
            push!(specs, TensorSpec("$root.scale_shift_table_a2v_ca_video",
                                    [5, d], ROW_MAJOR_SOURCE))
        end
    end

    push!(specs, TensorSpec(key("scale_shift_table"), [2, d],
                            ROW_MAJOR_SOURCE))
    _ltx_linear_specs!(specs, key("proj_out"), config.video_channels, d)
    if !config.video_only
        push!(specs, TensorSpec(key("audio_scale_shift_table"), [2, a],
                                ROW_MAJOR_SOURCE))
        _ltx_linear_specs!(specs, key("audio_proj_out"),
                           config.audio_channels, a)
    end
    specs
end
