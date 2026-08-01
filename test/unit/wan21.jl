@testset "Wan 2.1 configuration and mapping inventory" begin
    small = Wan21Config(hidden_size=8, ffn_size=16, frequency_size=4,
        text_size=12, heads=2, layers=2)
    specs = wan21_transformer_specs(small)
    @test length(specs) == 69
    @test length(unique(spec.source_key for spec in specs)) == length(specs)
    @test only(filter(s -> s.source_key == "patch_embedding.weight", specs)).source_shape ==
        [8, 16, 1, 2, 2]
    @test only(filter(s -> s.source_key == "blocks.1.ffn.0.weight", specs)).source_shape ==
        [16, 8]
    config = wan21_config(:t2v_1_3b)
    @test (config.hidden_size, config.ffn_size, config.heads, config.layers) ==
        (1536, 8960, 12, 30)
    @test length(wan21_transformer_specs(config)) == 825
    config14 = wan21_config(:t2v_14b)
    @test (config14.hidden_size, config14.ffn_size,
           config14.heads, config14.layers) ==
        (5120, 13824, 40, 40)
    @test length(wan21_transformer_specs(config14)) == 1095
    i2v = wan21_config(:i2v_14b_480p)
    @test (i2v.model_type, i2v.input_channels,
           i2v.output_channels) == (:i2v, 36, 16)
    @test length(wan21_transformer_specs(i2v)) == 1303
    @test model_family(Wan21()) == :wan21
    @test length(lora_targets(Wan21(), nothing, :attention_and_ffn)) == 2
    @test_throws ArgumentError wan21_config(:unknown)
end
