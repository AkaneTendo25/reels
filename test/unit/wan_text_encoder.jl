@testset "Wan UMT5 text encoder" begin
    config = UMT5Config(vocab_size=11, hidden_size=4, attention_size=4,
        ffn_size=6, heads=2, layers=1, buckets=8)
    specs = umt5_encoder_specs(config)
    @test length(specs) == 12
    @test first(specs).source_key == "token_embedding.weight"
    @test last(specs).source_key == "norm.weight"

    buckets = t5_relative_buckets(3, 3, 8)
    @test size(buckets) == (3, 3)
    @test all(buckets[index, index] == 1 for index in axes(buckets, 1))
    @test buckets[1, 2] == 6
    @test buckets[2, 1] == 2

    position_embedding = reshape(Float32.(1:16), 2, 8)
    bias = t5_position_bias(position_embedding, 3, 3)
    @test size(bias) == (3, 3, 2, 1)
    @test bias[2, 1, 1, 1] == position_embedding[1, buckets[1, 2]]

    norm = RMSNorm(Float32[1, 2, 3, 4], 1f-6)
    normalized = t5_rmsnorm(norm, reshape(Float32[1, 2, 3, 4], 4, 1, 1))
    @test size(normalized) == (4, 1, 1)

    dense(out, input, value) =
        DenseLayer(fill(Float32(value), out, input), nothing)
    attention = T5Attention(dense(4, 4, 0.03), dense(4, 4, -0.02),
        dense(4, 4, 0.04), dense(4, 4, 0.05), 2)
    feed_forward = T5FeedForward(dense(6, 4, 0.02),
        dense(6, 4, 0.03), dense(4, 6, 0.04))
    block = T5EncoderBlock(RMSNorm(4), attention, RMSNorm(4),
        feed_forward, zeros(Float32, 2, 8))
    embedding = reshape(Float32.(1:44) ./ 50f0, 4, 11)
    model = UMT5Encoder(config, embedding, [block], RMSNorm(4), nothing)
    ids = Int32[0 3; 1 4; 2 5]
    mask = BitMatrix(Bool[1 1; 1 1; 0 1])
    output = umt5_forward(model, ids; mask=mask)
    @test size(output) == (4, 3, 2)
    @test all(isfinite, output)
    @test output == umt5_forward(model, ids; mask=mask)

    mktempdir() do directory
        checkpoint = joinpath(directory, "umt5.safetensors")
        state = umt5_encoder_state_dict(model)
        write_safetensors(checkpoint, state)
        loaded = load_umt5_encoder(checkpoint, config)
        @test isempty(audit_state_dict(open_tensor_source(checkpoint), specs))
        @test umt5_encoder_state_dict(loaded) == state
        @test umt5_forward(loaded, ids; mask=mask) == output

        quantized_path = joinpath(directory, "umt5-int8.safetensors")
        write_quantized_umt5(quantized_path, model; compute_type=Float32)
        header = inspect_safetensors(quantized_path)
        @test header.metadata["format"] == "reels-umt5-int8-v1"
        @test header.tensors["blocks.0.attn.q.weight"].dtype == "I8"
        @test header.tensors["blocks.0.attn.q.weight.scale"].dtype == "F32"
        loaded_int8 = load_quantized_umt5_encoder(
            quantized_path, config; compute_type=Float32)
        @test loaded_int8.blocks[1].attention.q.weight isa QuantizedMatrix
        int8_output = umt5_forward(loaded_int8, ids; mask=mask)
        @test int8_output ≈ output rtol=3f-2 atol=3f-2

        transfer = frozen_weight_transfer(
            :cpu, :fp32; quantization=:int8)
        quantized_on_load = move_to_device(loaded, transfer)
        @test quantized_on_load.blocks[1].attention.q.weight isa
            QuantizedMatrix
        @test quantized_on_load.token_embedding isa Matrix{Float32}
        @test umt5_forward(quantized_on_load, ids; mask=mask) ≈
            output rtol=3f-2 atol=3f-2

        fp16_transfer = frozen_weight_transfer(
            :cpu, :fp16; quantization=:int8)
        fp16_quantized = move_to_device(loaded, fp16_transfer)
        fp16_output = umt5_forward(fp16_quantized, ids; mask=mask)
        @test eltype(fp16_quantized.token_embedding) === Float16
        @test eltype(fp16_output) === Float32
        @test all(isfinite, fp16_output)
        @test fp16_output ≈ output rtol=4f-2 atol=4f-2
    end

    official = UMT5Config()
    @test length(umt5_encoder_specs(official)) == 242
    @test Reels._tokenizer_vocabulary_fits(256_000, 256_384)
    @test Reels._tokenizer_vocabulary_fits(256_384, 256_384)
    @test !Reels._tokenizer_vocabulary_fits(256_385, 256_384)
end
