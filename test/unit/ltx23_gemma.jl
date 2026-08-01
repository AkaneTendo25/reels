@testset "native Gemma-3 text encoder for LTX-2.3" begin
    official = Gemma3TextConfig()
    @test (official.hidden_size, official.intermediate_size,
           official.layers, official.heads, official.kv_heads,
           official.head_dim) == (3840, 15360, 48, 16, 8, 256)
    @test length(gemma3_text_encoder_specs(official)) == 626
    official_specs = Dict(spec.source_key => spec.source_shape
        for spec in gemma3_text_encoder_specs(official))
    @test official_specs[
        "language_model.model.embed_tokens.weight"] == [262208, 3840]
    @test official_specs[
        "language_model.model.layers.47.self_attn.q_proj.weight"] ==
        [4096, 3840]
    @test official_specs[
        "language_model.model.layers.47.self_attn.k_proj.weight"] ==
        [2048, 3840]
    @test official_specs[
        "language_model.model.layers.47.mlp.down_proj.weight"] ==
        [3840, 15360]

    parsed = gemma3_text_config(Dict("text_config" => Dict(
        "vocab_size" => 16, "hidden_size" => 8,
        "intermediate_size" => 16, "num_hidden_layers" => 2,
        "num_attention_heads" => 2, "num_key_value_heads" => 1,
        "head_dim" => 4, "rms_norm_eps" => 1e-6,
        "rope_theta" => 1000, "rope_local_base_freq" => 100,
        "rope_scaling" => Dict("factor" => 2),
        "sliding_window" => 4, "sliding_window_pattern" => 2,
        "query_pre_attn_scalar" => 4,
        "max_position_embeddings" => 4,
        "ltx_max_length" => 4)))
    @test parsed.max_length == 4
    @test length(gemma3_text_encoder_specs(parsed)) == 28

    norm = Gemma3Norm(zeros(Float32, 4), 1f-6)
    normalized = gemma3_norm(norm,
        reshape(Float32[1, 2, 3, 4], 4, 1, 1))
    @test sum(abs2, normalized) ≈ 4f0 rtol=2f-6

    rng = Xoshiro(250)
    model = Gemma3TextEncoder(parsed; rng=rng)
    ids = Int32[0 0; 1 0; 2 3; 4 5]
    mask = Bool[false false; true false; true true; true true]
    states = gemma3_forward(model, ids, mask)
    @test length(states) == 3
    @test all(state -> size(state) == (8, 4, 2), states)
    @test all(state -> all(isfinite, state), states)
    @test all(isapprox(sum(abs2, states[end][:, token, batch]), 8f0;
                       rtol=2f-5)
              for token in 1:4, batch in 1:2)

    local_output = gemma3_block_forward(model.blocks[1], parsed,
        states[1], mask, 1)
    global_output = gemma3_block_forward(model.blocks[2], parsed,
        states[1], mask, 2)
    @test size(local_output) == size(global_output) == (8, 4, 2)
    @test local_output != global_output

    mktempdir() do directory
        specs = gemma3_text_encoder_specs(parsed)
        state = Dict{String,AbstractArray}(
            spec.source_key => zeros(Float32, spec.source_shape...)
            for spec in specs)
        state["language_model.model.embed_tokens.weight"] .=
            reshape(Float32.(1:(16 * 8)), 16, 8) ./ 100
        path = joinpath(directory, "tiny-gemma.safetensors")
        write_safetensors(path, state)
        source = open_tensor_source(path)
        @test isempty(audit_state_dict(source, specs))
        loaded = load_gemma3_text_encoder(source, parsed; strict=true)
        loaded_states = gemma3_forward(loaded, ids, mask)
        @test length(loaded_states) == 3
        @test all(isfinite, loaded_states[end])
        @test loaded_states[1] == loaded_states[2]
    end

    @test_throws ArgumentError Gemma3TextConfig(heads=3, kv_heads=2)
    @test_throws DimensionMismatch gemma3_forward(model, ids[1:3, :], mask)
end
