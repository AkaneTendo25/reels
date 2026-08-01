@testset "video manifest, buckets, and preprocessing cache" begin
    mktempdir() do dir
        clips = joinpath(dir, "clips")
        mkpath(clips)
        write(joinpath(clips, "a.mp4"), UInt8[1, 2, 3])
        write(joinpath(clips, "b.mp4"), UInt8[4, 5, 6])
        write(joinpath(clips, "b.txt"), "second caption\n")
        manifest = joinpath(dir, "dataset.toml")
        write(manifest, """
        [[samples]]
        video = "clips/a.mp4"
        caption = "first caption"
        [[samples]]
        video = "clips/b.mp4"
        caption_file = "clips/b.txt"
        """)
        samples = load_video_manifest(manifest; caption_prefix="trigger, ")
        @test length(samples) == 2
        @test samples[1].caption == "trigger, first caption"
        @test samples[2].caption == "trigger, second caption"
        @test isabspath(samples[1].video)
        @test samples[1].id != samples[2].id

        jsonl = joinpath(dir, "dataset.jsonl")
        write(jsonl,
            "{\"video\":\"clips/a.mp4\",\"caption\":\"json caption\"}\n")
        @test only(load_video_manifest(jsonl)).caption == "json caption"

        bucket = assign_video_bucket(
            VideoMetadata(1920, 1080, 100, 25.0),
            [(512, 512), (768, 432)], [17, 33, 49]; target_fps=16)
        @test (bucket.width, bucket.height, bucket.frames, bucket.fps) ==
            (768, 432, 49, 16)
        @test (bucket.resized_width, bucket.resized_height,
               bucket.crop_x, bucket.crop_y) == (768, 432, 0, 0)
        @test_throws ArgumentError assign_video_bucket(
            VideoMetadata(10, 10, 2, 30.0), [(8, 8)], [17]; target_fps=16)
        @test_throws ArgumentError assign_video_bucket(
            VideoMetadata(10, 10, 100, 30.0), [(8, 8)], [16]; target_fps=16)

        identity = PreprocessIdentity(model_family="wan21",
            model_checkpoint="transformer-sha", text_encoder_checkpoint="t5-sha",
            vae_checkpoint="vae-sha", vae_scale="0.18215", dtype="Float32")
        key = preprocess_cache_key(samples[1], bucket, identity)
        @test length(key) == 64
        changed_sample = VideoSample(samples[1].id, samples[1].video,
                                     samples[1].caption * " changed")
        @test preprocess_cache_key(changed_sample, bucket, identity) != key

        cache_dir = joinpath(dir, "cache")
        path = cache_entry_path(cache_dir, key)
        latents = reshape(Float32.(1:16), 1, 2, 2, 4)
        context = reshape(Float32.(1:15), 5, 3)
        write_preprocess_cache(path, key, latents, context;
            metadata=Dict("sample_id" => samples[1].id))
        @test cache_is_valid(path, key)
        @test !cache_is_valid(path, "0"^64)
        loaded = load_preprocess_cache(path; expected_key=key)
        @test loaded.latents == latents
        @test loaded.text_context == context
        @test loaded.metadata["sample_id"] == samples[1].id
        @test !isfile(path * ".tmp")

        entries = String[]
        for index in 1:3
            entry_key = string(index)^64
            entry_path = cache_entry_path(cache_dir, entry_key)
            write_preprocess_cache(entry_path, entry_key,
                fill(Float32(index), 1, 2, 2, 4),
                fill(Float32(index), 5, 3))
            push!(entries, entry_path)
        end
        provider = CachedBatchProvider(entries; batch_size=2)
        first_batch = provider(Xoshiro(80))
        repeated = provider(Xoshiro(80))
        @test first_batch isa WanLatentBatch
        @test size(first_batch.latents) == (1, 2, 2, 4, 2)
        @test size(first_batch.text_context) == (5, 3, 2)
        @test first_batch.latents == repeated.latents
        @test first_batch.text_context == repeated.text_context
    end
end

@testset "Wan I2V preprocessing cache and provider" begin
    mktempdir() do directory
        entries = String[]
        for index in 1:2
            key = string(index + 3)^64
            path = cache_entry_path(directory, key)
            write_preprocess_cache(
                path, key,
                fill(Float32(index), 16, 2, 2, 4),
                fill(Float32(index), 6, 3);
                conditioning_video=
                    fill(Float32(index), 20, 2, 2, 4),
                image_features=
                    fill(Float32(index), 1280, 5))
            @test inspect_preprocess_cache(path).valid
            loaded = load_preprocess_cache(path)
            @test size(loaded.conditioning_video) == (20, 2, 2, 4)
            @test size(loaded.image_features) == (1280, 5)
            push!(entries, path)
        end
        batch = CachedBatchProvider(entries; batch_size=2)(Xoshiro(91))
        @test size(batch.latents) == (16, 2, 2, 4, 2)
        @test size(batch.conditioning_video) == (20, 2, 2, 4, 2)
        @test size(batch.image_features) == (1280, 5, 2)
    end
end

@testset "resumable Wan latent cache stage" begin
    mktempdir() do directory
        path = joinpath(directory, "entry.safetensors.latents-stage")
        key = "a"^64
        latents = reshape(Float32.(1:24), 2, 3, 2, 2)
        Reels._write_wan_latent_stage(path, key, latents)
        @test Reels._load_wan_latent_stage(path, key) == latents
        @test Reels._load_wan_latent_stage(path, "b"^64) === nothing
        write(path, "corrupt")
        @test Reels._load_wan_latent_stage(path, key) === nothing
    end
end

@testset "checkpoint fingerprints" begin
    mktemp() do path, stream
        write(stream, "reels-checkpoint")
        close(stream)
        first_fingerprint = checkpoint_fingerprint(path)
        @test length(first_fingerprint) == 64
        @test checkpoint_fingerprint(path) == first_fingerprint
    end
end

@testset "distributed cache sampling is deterministic and disjoint" begin
    rank0_rng = Xoshiro(905)
    rank1_rng = Xoshiro(905)
    first_rank = distributed_sample_indices(rank0_rng, 12, 3, 0, 2)
    second_rank = distributed_sample_indices(rank1_rng, 12, 3, 1, 2)
    @test isempty(intersect(first_rank, second_rank))
    @test length(first_rank) == length(second_rank) == 3
    @test distributed_sample_indices(Xoshiro(905), 12, 3, 0, 2) ==
        first_rank
    @test_throws ArgumentError distributed_sample_indices(
        Xoshiro(1), 5, 3, 0, 2)
    @test_throws ArgumentError distributed_sample_indices(
        Xoshiro(1), 12, 3, 2, 2)
end

@testset "cached validation conditioning" begin
    mktempdir() do directory
        wan_path = joinpath(directory, "wan-validation.safetensors")
        wan_noise = reshape(Float32.(1:8), 1, 1, 2, 4, 1)
        wan_context = reshape(Float32.(1:10), 5, 2, 1)
        write_safetensors(wan_path, Dict(
            "noise" => wan_noise,
            "text_context" => wan_context,
        ); metadata=Dict(
            "format" => "reels-validation-cache-v1",
            "cache_key" => "wan-test",
            "model_family" => "wan21",
            "prompt" => "A test wave.",
            "seed" => "42",
        ))
        @test inspect_validation_cache(wan_path).valid
        wan = load_validation_cache(wan_path)
        @test wan isa WanValidationBatch
        @test wan.prompt == "A test wave."
        @test wan.noise == wan_noise
        @test wan.text_context == wan_context
        @test wan.conditioning_video === nothing

        i2v_path = joinpath(directory, "wan-i2v-validation.safetensors")
        i2v_conditioning = zeros(Float32, 20, 1, 2, 4, 1)
        i2v_features = zeros(Float32, 1280, 3, 1)
        write_safetensors(i2v_path, Dict(
            "noise" => reshape(zeros(Float32, 16 * 8), 16, 1, 2, 4, 1),
            "text_context" => wan_context,
            "conditioning_video" => i2v_conditioning,
            "image_features" => i2v_features,
        ); metadata=Dict(
            "format" => "reels-validation-cache-v1",
            "cache_key" => "wan-i2v-test",
            "model_family" => "wan21",
            "prompt" => "An I2V test.",
            "seed" => "42",
        ))
        @test inspect_validation_cache(i2v_path).valid
        i2v = load_validation_cache(i2v_path)
        @test i2v.conditioning_video == i2v_conditioning
        @test i2v.image_features == i2v_features

        ltx_path = joinpath(directory, "ltx-validation.safetensors")
        ltx_noise = reshape(Float32.(1:6), 3, 2, 1)
        ltx_context = reshape(Float32.(1:8), 4, 2, 1)
        positions = zeros(Float32, 3, 2, 1)
        write_safetensors(ltx_path, Dict(
            "noise" => ltx_noise,
            "text_context" => ltx_context,
            "positions" => positions,
        ); metadata=Dict(
            "format" => "reels-validation-cache-v1",
            "cache_key" => "ltx-test",
            "model_family" => "ltx23",
            "prompt" => "A test pan.",
            "seed" => "42",
        ))
        @test inspect_validation_cache(ltx_path).valid
        ltx = load_validation_cache(ltx_path)
        @test ltx isa LTXValidationBatch
        @test ltx.prompt == "A test pan."
        @test ltx.noise == ltx_noise
        @test ltx.positions == positions
    end
end
