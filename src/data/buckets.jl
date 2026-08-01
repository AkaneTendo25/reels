struct VideoMetadata
    width::Int
    height::Int
    frames::Int
    fps::Float64
    duration::Float64
end
VideoMetadata(width::Integer, height::Integer, frames::Integer, fps::Real) =
    VideoMetadata(Int(width), Int(height), Int(frames), Float64(fps),
                  Float64(frames) / Float64(fps))

struct BucketAssignment
    width::Int
    height::Int
    frames::Int
    fps::Int
    resized_width::Int
    resized_height::Int
    crop_x::Int
    crop_y::Int
end

function assign_video_bucket(metadata::VideoMetadata,
                             resolution_buckets,
                             frame_buckets;
                             target_fps::Integer,
                             temporal_multiple::Integer=4,
                             temporal_offset::Integer=1)
    metadata.width > 0 && metadata.height > 0 && metadata.frames > 0 ||
        throw(ArgumentError("video dimensions and frame count must be positive"))
    metadata.fps > 0 || throw(ArgumentError("video FPS must be positive"))
    target_fps > 0 || throw(ArgumentError("target FPS must be positive"))
    isempty(resolution_buckets) &&
        throw(ArgumentError("resolution bucket list is empty"))
    isempty(frame_buckets) && throw(ArgumentError("frame bucket list is empty"))
    valid_frames = sort!(unique(Int.(frame_buckets)))
    for frames in valid_frames
        frames > 0 || throw(ArgumentError("frame buckets must be positive"))
        (frames - temporal_offset) % temporal_multiple == 0 ||
            throw(ArgumentError("frame bucket $frames is incompatible with " *
                "temporal form $(temporal_multiple)n+$temporal_offset"))
    end
    available = floor(Int, metadata.frames * target_fps / metadata.fps)
    candidates = filter(frames -> frames <= available, valid_frames)
    isempty(candidates) &&
        throw(ArgumentError("video provides $available frames at $target_fps FPS; " *
            "smallest configured bucket is $(first(valid_frames))"))
    frames = last(candidates)

    source_aspect = metadata.width / metadata.height
    resolutions = [(Int(width), Int(height)) for
                   (width, height) in resolution_buckets]
    all(pair -> pair[1] > 0 && pair[2] > 0, resolutions) ||
        throw(ArgumentError("resolution buckets must be positive"))
    scores = [(abs(log(source_aspect / (width / height))),
               -width * height, index) for
              (index, (width, height)) in enumerate(resolutions)]
    _, _, selected = minimum(scores)
    width, height = resolutions[selected]
    scale = max(width / metadata.width, height / metadata.height)
    resized_width = max(width, ceil(Int, metadata.width * scale))
    resized_height = max(height, ceil(Int, metadata.height * scale))
    crop_x = (resized_width - width) ÷ 2
    crop_y = (resized_height - height) ÷ 2
    BucketAssignment(width, height, frames, Int(target_fps),
        resized_width, resized_height, crop_x, crop_y)
end
