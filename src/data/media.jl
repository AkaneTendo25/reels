function ffmpeg_executable()
    path = Sys.which("ffmpeg")
    path === nothing &&
        throw(ArgumentError("ffmpeg executable was not found on PATH"))
    path
end
function ffprobe_executable()
    path = Sys.which("ffprobe")
    path === nothing &&
        throw(ArgumentError("ffprobe executable was not found on PATH"))
    path
end

function _run_captured(command::Cmd, operation)
    stdout_buffer, stderr_buffer = IOBuffer(), IOBuffer()
    process = run(pipeline(ignorestatus(command),
        stdout=stdout_buffer, stderr=stderr_buffer))
    output = take!(stdout_buffer)
    diagnostic = strip(String(take!(stderr_buffer)))
    success(process) || throw(ArgumentError(
        "$operation failed with exit code $(process.exitcode)" *
        (isempty(diagnostic) ? "" : ": $diagnostic")))
    output
end

function _run_captured_input(command::Cmd, input::Vector{UInt8}, operation)
    stdout_buffer, stderr_buffer = IOBuffer(), IOBuffer()
    process = run(pipeline(ignorestatus(command),
        stdin=IOBuffer(input), stdout=stdout_buffer, stderr=stderr_buffer))
    diagnostic = strip(String(take!(stderr_buffer)))
    success(process) || throw(ArgumentError(
        "$operation failed with exit code $(process.exitcode)" *
        (isempty(diagnostic) ? "" : ": $diagnostic")))
    nothing
end

function _parse_frame_rate(value)
    text = string(value)
    text in ("N/A", "0/0", "") && return 0.0
    parts = split(text, '/')
    if length(parts) == 2
        denominator = parse(Float64, parts[2])
        denominator == 0 && return 0.0
        return parse(Float64, parts[1]) / denominator
    end
    parse(Float64, text)
end

function _parse_optional_float(value, fallback=0.0)
    value === nothing && return fallback
    text = string(value)
    text in ("N/A", "") ? fallback : parse(Float64, text)
end

"""
    probe_video(path; ffprobe=ffprobe_executable())

Inspect the first video stream without decoding frames. Returns validated
dimensions, frame count, average FPS, and duration.
"""
function probe_video(path::AbstractString; ffprobe=ffprobe_executable())
    isfile(path) || throw(ArgumentError("video does not exist: $path"))
    command = Cmd([String(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,duration:format=duration",
        "-of", "json", String(path)])
    raw = _run_captured(command, "ffprobe for $path")
    document = try
        parse_json(String(raw))
    catch error
        throw(ArgumentError("ffprobe returned invalid JSON for $path: " *
            sprint(showerror, error)))
    end
    streams = get(document, "streams", Any[])
    isempty(streams) &&
        throw(ArgumentError("video has no decodable video stream: $path"))
    stream = first(streams)
    width = Int(stream["width"])
    height = Int(stream["height"])
    fps = _parse_frame_rate(get(stream, "avg_frame_rate", "0/0"))
    duration = _parse_optional_float(get(stream, "duration", nothing),
        _parse_optional_float(get(get(document, "format", Dict()),
                                  "duration", nothing), 0.0))
    frames_text = string(get(stream, "nb_frames", "N/A"))
    frames = frames_text in ("N/A", "") ?
        floor(Int, duration * fps + 1e-6) : parse(Int, frames_text)
    width > 0 && height > 0 ||
        throw(ArgumentError("video reports invalid dimensions: $(width)x$(height)"))
    fps > 0 || throw(ArgumentError("video reports invalid average FPS"))
    frames > 0 ||
        throw(ArgumentError("video reports no frames and duration fallback failed"))
    duration <= 0 && (duration = frames / fps)
    VideoMetadata(width, height, frames, fps, duration)
end

function _normalization(values, mode)
    floats = Float32.(values)
    mode === :zero_one && return floats ./ 255f0
    mode === :minus_one_one && return floats ./ 127.5f0 .- 1f0
    throw(ArgumentError("normalization must be :zero_one or :minus_one_one"))
end

"""
    decode_video_frames(path, metadata, bucket; temporal_crop=:center)

Decode exactly one bucket as `(channels,frames,height,width)` Float32 RGB.
FFmpeg applies one fixed resize/crop transform to the entire clip.
"""
function decode_video_frames(path::AbstractString, metadata::VideoMetadata,
                             bucket::BucketAssignment;
                             temporal_crop=:center,
                             normalization=:minus_one_one,
                             ffmpeg=ffmpeg_executable())
    temporal_crop in (:start, :center) ||
        throw(ArgumentError("temporal_crop must be :start or :center"))
    clip_duration = bucket.frames / bucket.fps
    start_time = temporal_crop === :center ?
        max(0.0, (metadata.duration - clip_duration) / 2) : 0.0
    filter = "trim=start=$(start_time),setpts=PTS-STARTPTS," *
        "fps=$(bucket.fps)," *
        "scale=$(bucket.resized_width):$(bucket.resized_height):flags=bicubic," *
        "crop=$(bucket.width):$(bucket.height):$(bucket.crop_x):$(bucket.crop_y)"
    command = Cmd([String(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-noautorotate", "-i", String(path),
        "-vf", filter, "-frames:v", string(bucket.frames),
        "-an", "-sn", "-dn", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    bytes = _run_captured(command, "ffmpeg decode for $path")
    expected = 3 * bucket.width * bucket.height * bucket.frames
    length(bytes) == expected ||
        throw(ArgumentError("ffmpeg decoded $(length(bytes)) RGB bytes for $path; " *
            "expected $expected ($(bucket.frames) frames at " *
            "$(bucket.width)x$(bucket.height))"))
    raw = reshape(bytes, 3, bucket.width, bucket.height, bucket.frames)
    frames = permutedims(raw, (1, 4, 3, 2))
    _normalization(frames, normalization)
end

function decode_video_sample(sample::VideoSample, resolution_buckets,
                             frame_buckets; target_fps,
                             temporal_crop=:center,
                             normalization=:minus_one_one,
                             ffmpeg=ffmpeg_executable(),
                             ffprobe=ffprobe_executable())
    metadata = probe_video(sample.video; ffprobe=ffprobe)
    bucket = assign_video_bucket(metadata, resolution_buckets, frame_buckets;
                                 target_fps=target_fps)
    frames = decode_video_frames(sample.video, metadata, bucket;
        temporal_crop=temporal_crop, normalization=normalization,
        ffmpeg=ffmpeg)
    (sample=sample, metadata=metadata, bucket=bucket, frames=frames)
end

"""
    write_video_frames(path, frames; fps=16, normalization=:minus_one_one)

Encode host or device RGB frames with layout `(channels,frames,height,width)`
to an H.264 MP4 through FFmpeg. `normalization` accepts `:minus_one_one` or
`:zero_one`; values outside the selected range are clamped before conversion
to RGB24.
"""
function write_video_frames(path::AbstractString, frames::AbstractArray;
                            fps::Real=16,
                            normalization::Symbol=:minus_one_one,
                            ffmpeg=ffmpeg_executable())
    ndims(frames) == 4 ||
        throw(DimensionMismatch(
            "video frames must be (channels,frames,height,width)"))
    channels, count, height, width = size(frames)
    channels == 3 ||
        throw(DimensionMismatch("video frames must contain three RGB channels"))
    count > 0 && height > 0 && width > 0 ||
        throw(DimensionMismatch("video frame dimensions must be positive"))
    isfinite(fps) && fps > 0 ||
        throw(ArgumentError("video FPS must be finite and positive"))
    normalization in (:minus_one_one, :zero_one) ||
        throw(ArgumentError(
            "normalization must be :minus_one_one or :zero_one"))

    host = Array(float32_values(frames))
    scaled = normalization === :minus_one_one ?
        (host .+ 1f0) .* 127.5f0 : host .* 255f0
    pixels = UInt8.(round.(clamp.(scaled, 0f0, 255f0)))
    # Raw RGB24 requires channel-fast, then width, height, and frame order.
    bytes = vec(permutedims(pixels, (1, 4, 3, 2)))
    destination = abspath(String(path))
    mkpath(dirname(destination))
    command = Cmd([String(ffmpeg), "-nostdin", "-hide_banner",
        "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "$(width)x$(height)",
        "-framerate", string(Float64(fps)),
        "-i", "pipe:0", "-frames:v", string(count),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", destination])
    _run_captured_input(
        command, bytes, "ffmpeg encode for $destination")
    isfile(destination) && filesize(destination) > 0 ||
        throw(ErrorException("ffmpeg did not create video: $destination"))
    destination
end
