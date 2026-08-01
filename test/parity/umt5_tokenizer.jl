@testset "official UMT5 SentencePiece parity" begin
    model_path = get(ENV, "REELS_TEST_UMT5_TOKENIZER", "")
    if isempty(model_path)
        @test_skip "set REELS_TEST_UMT5_TOKENIZER to the official spiece.model"
    else
        tokenizer = SentencePieceTokenizer(model_path)
        cases = [
            "A red fox jumps over snow." =>
                Int32[320, 4062, 273, 56209, 48150, 281, 702, 45540, 274,
                      1, 0, 0],
            "Привет, мир!" =>
                Int32[201698, 275, 13337, 332, 1, 0, 0, 0, 0, 0, 0, 0],
            "猫が窓辺に座っている" =>
                Int32[273, 14985, 409, 19810, 24657, 435, 5398, 7937,
                      1, 0, 0, 0],
        ]
        for (text, expected) in cases
            tokenized = tokenize_wan(tokenizer, text; max_length=12)
            @test tokenized.ids[:, 1] == expected
            expected_length = findfirst(==(Int32(1)), expected)
            @test tokenized.lengths == [expected_length]
            @test tokenized.mask[:, 1] ==
                BitVector(eachindex(expected) .<= expected_length)
        end

        tiny_config = UMT5Config(vocab_size=tokenizer.vocab_size,
            hidden_size=4, attention_size=4, ffn_size=6, heads=2,
            layers=1, buckets=8)
        dense(output, input) =
            DenseLayer(zeros(Float32, output, input), nothing)
        attention = T5Attention(dense(4, 4), dense(4, 4), dense(4, 4),
            dense(4, 4), 2)
        feed_forward =
            T5FeedForward(dense(6, 4), dense(6, 4), dense(4, 6))
        block = T5EncoderBlock(RMSNorm(4), attention, RMSNorm(4),
            feed_forward, zeros(Float32, 2, 8))
        encoder = UMT5Encoder(tiny_config,
            zeros(Float32, 4, tokenizer.vocab_size), [block],
            RMSNorm(4), nothing)
        conditioner = WanTextConditioner(encoder, tokenizer, 12, :cpu)
        encoded = encode_text(Wan21(), conditioner, first(cases).first)
        @test size(encoded.context) == (4, 12, 1)
        @test encoded.lengths == [10]
        @test all(iszero, encoded.context[:, 11:12, 1])

        mktempdir() do directory
            video_path = joinpath(directory, "clip.mkv")
            run(Cmd([ffmpeg_executable(), "-nostdin", "-hide_banner",
                "-loglevel", "error", "-f", "lavfi", "-i",
                "testsrc2=size=8x8:rate=4:duration=2", "-c:v", "ffv1",
                "-y", video_path]))
            sample = VideoSample("clip", video_path,
                "A red fox jumps over snow.")
            vae_config = WanVAEConfig(base_channels=2, latent_channels=16,
                channel_multipliers=[1, 2], residual_blocks=1,
                temporal_downsample=[true])
            vae = WanVAEEncoder(vae_config; rng=Xoshiro(121))
            identity = PreprocessIdentity(model_family="wan21",
                model_checkpoint="tiny", text_encoder_checkpoint="tiny-text",
                vae_checkpoint="tiny-vae", vae_scale="wan21",
                dtype="fp32")
            cached = build_wan_preprocess_cache(directory, Wan21(),
                conditioner, vae, sample, identity, [(8, 8)], [5];
                target_fps=4)
            @test cached.created
            entry = load_preprocess_cache(cached.path;
                expected_key=cached.key)
            @test size(entry.latents) == (16, 3, 4, 4)
            @test size(entry.text_context) == (4, 12)
            repeated = build_wan_preprocess_cache(directory, Wan21(),
                conditioner, vae, sample, identity, [(8, 8)], [5];
                target_fps=4)
            @test !repeated.created
        end
        close(tokenizer)
    end
end
