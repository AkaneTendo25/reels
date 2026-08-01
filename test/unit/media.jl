@testset "native FFmpeg video probing and decoding" begin
    ffmpeg = ffmpeg_executable()
    ffprobe = ffprobe_executable()
    mktempdir() do dir
        video = joinpath(dir, "reference.mkv")
        command = Cmd([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x32:rate=8:duration=2",
            "-c:v", "ffv1", "-y", video])
        run(command)

        metadata = probe_video(video; ffprobe=ffprobe)
        @test metadata.width == 64
        @test metadata.height == 32
        @test metadata.frames == 16
        @test metadata.fps ≈ 8
        @test metadata.duration ≈ 2 atol=0.02

        bucket = assign_video_bucket(metadata, [(32, 32)], [5];
                                     target_fps=4)
        @test (bucket.resized_width, bucket.resized_height,
               bucket.crop_x, bucket.crop_y) == (64, 32, 16, 0)
        centered = decode_video_frames(video, metadata, bucket;
            temporal_crop=:center, ffmpeg=ffmpeg)
        repeated = decode_video_frames(video, metadata, bucket;
            temporal_crop=:center, ffmpeg=ffmpeg)
        started = decode_video_frames(video, metadata, bucket;
            temporal_crop=:start, normalization=:zero_one, ffmpeg=ffmpeg)
        @test size(centered) == (3, 5, 32, 32)
        @test eltype(centered) == Float32
        @test extrema(centered)[1] >= -1f0
        @test extrema(centered)[2] <= 1f0
        @test centered == repeated
        @test extrema(started)[1] >= 0f0
        @test extrema(started)[2] <= 1f0
        @test centered != started .* 2f0 .- 1f0

        sample = VideoSample("reference", video, "test pattern")
        decoded = decode_video_sample(sample, [(32, 32)], [5];
            target_fps=4, ffmpeg=ffmpeg, ffprobe=ffprobe)
        @test decoded.sample === sample
        @test decoded.bucket == bucket
        @test decoded.frames == centered

        encoded = write_video_frames(
            joinpath(dir, "encoded.mp4"), centered; fps=4)
        encoded_metadata = probe_video(encoded; ffprobe=ffprobe)
        @test encoded_metadata.width == 32
        @test encoded_metadata.height == 32
        @test encoded_metadata.frames == 5
        @test encoded_metadata.fps ≈ 4

        corrupt = joinpath(dir, "corrupt.mp4")
        write(corrupt, "not a video")
        error = try
            probe_video(corrupt; ffprobe=ffprobe)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("ffprobe", sprint(showerror, error))
    end
end
