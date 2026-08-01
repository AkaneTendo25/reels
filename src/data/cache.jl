Base.@kwdef struct PreprocessIdentity
    model_family::String
    model_checkpoint::String
    text_encoder_checkpoint::String
    image_encoder_checkpoint::String = ""
    vae_checkpoint::String
    preprocessing_version::String = "1"
    vae_scale::String
    dtype::String
end

function checkpoint_fingerprint(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("checkpoint fingerprint requires a file: $path"))
    open(path, "r") do stream
        bytes2hex(sha256(stream))
    end
end

function source_media_fingerprint(path::AbstractString)
    isfile(path) || throw(ArgumentError("source media does not exist: $path"))
    info = stat(path)
    bytes2hex(sha256(json_encode(Dict(
        "path" => normpath(abspath(path)),
        "size" => Int(info.size),
        "mtime" => string(info.mtime),
    ))))
end

function preprocess_cache_key(sample::VideoSample, bucket::BucketAssignment,
                              identity::PreprocessIdentity)
    payload = Dict(
        "source" => source_media_fingerprint(sample.video),
        "caption" => sample.caption,
        "model_family" => identity.model_family,
        "model_checkpoint" => identity.model_checkpoint,
        "text_encoder_checkpoint" => identity.text_encoder_checkpoint,
        "image_encoder_checkpoint" => identity.image_encoder_checkpoint,
        "vae_checkpoint" => identity.vae_checkpoint,
        "preprocessing_version" => identity.preprocessing_version,
        "width" => bucket.width,
        "height" => bucket.height,
        "frames" => bucket.frames,
        "fps" => bucket.fps,
        "resized_width" => bucket.resized_width,
        "resized_height" => bucket.resized_height,
        "crop_x" => bucket.crop_x,
        "crop_y" => bucket.crop_y,
        "vae_scale" => identity.vae_scale,
        "dtype" => identity.dtype,
    )
    bytes2hex(sha256(json_encode(payload)))
end

cache_entry_path(cache_dir::AbstractString, key::AbstractString) =
    joinpath(cache_dir, key[1:2], "$key.safetensors")

_wan_latent_stage_path(path::AbstractString) = path * ".latents-stage"

function _load_wan_latent_stage(path::AbstractString, key::AbstractString)
    isfile(path) || return nothing
    header = try
        inspect_safetensors(path)
    catch
        return nothing
    end
    get(header.metadata, "format", "") ==
        "reels-wan-latent-stage-v1" || return nothing
    get(header.metadata, "cache_key", "") == key || return nothing
    sort!(collect(keys(header.tensors))) == ["latents"] || return nothing
    shape = header.tensors["latents"].shape
    length(shape) == 4 || return nothing
    load_state_tensor(open_tensor_source(path),
        TensorSpec("latents", shape, ROW_MAJOR_SOURCE))
end

function _write_wan_latent_stage(path::AbstractString, key::AbstractString,
                                 latents::AbstractArray)
    ndims(latents) == 4 ||
        throw(DimensionMismatch(
            "staged Wan latents must be (channels,frames,height,width)"))
    write_safetensors(path, Dict("latents" => Array(latents)); metadata=Dict(
        "format" => "reels-wan-latent-stage-v1",
        "cache_key" => String(key)))
end

function write_preprocess_cache(path::AbstractString, key::AbstractString,
                                latents::AbstractArray,
                                text_context::AbstractArray;
                                metadata=Dict{String,String}(),
                                conditioning_video=nothing,
                                image_features=nothing)
    ndims(latents) == 4 ||
        throw(DimensionMismatch("cached latents must be (channels,frames,height,width)"))
    ndims(text_context) == 2 ||
        throw(DimensionMismatch("cached text context must be (features,tokens)"))
    (conditioning_video === nothing) ==
        (image_features === nothing) ||
        throw(ArgumentError(
            "Wan I2V cache requires both conditioning_video and image_features"))
    i2v = conditioning_video !== nothing
    if i2v
        ndims(conditioning_video) == 4 ||
            throw(DimensionMismatch(
                "I2V conditioning must be (channels,frames,height,width)"))
        size(conditioning_video)[2:end] == size(latents)[2:end] ||
            throw(DimensionMismatch(
                "I2V conditioning and latent dimensions differ"))
        ndims(image_features) == 2 ||
            throw(DimensionMismatch(
                "I2V image features must be (1280,tokens)"))
        size(image_features, 1) == 1280 ||
            throw(DimensionMismatch(
                "I2V image features must have 1280 channels"))
    end
    info = Dict{String,String}(
        "format" => i2v ?
            "reels-wan-i2v-preprocess-cache-v1" :
            "reels-preprocess-cache-v1",
        "cache_key" => String(key),
    )
    merge!(info, metadata)
    tensors = Dict{String,AbstractArray}(
        "latents" => Array(latents),
        "text_context" => Array(text_context),
    )
    if i2v
        tensors["conditioning_video"] = Array(conditioning_video)
        tensors["image_features"] = Array(image_features)
    end
    write_safetensors(path, tensors; metadata=info)
end

function inspect_preprocess_cache(path::AbstractString; expected_key=nothing)
    header = inspect_safetensors(path)
    errors = String[]
    format = get(header.metadata, "format", "")
    format in ("reels-preprocess-cache-v1",
               "reels-wan-i2v-preprocess-cache-v1") ||
        push!(errors, "unsupported cache format")
    if expected_key !== nothing &&
       get(header.metadata, "cache_key", "") != expected_key
        push!(errors, "cache key mismatch")
    end
    expected_tensors = format == "reels-wan-i2v-preprocess-cache-v1" ?
        ["conditioning_video", "image_features", "latents", "text_context"] :
        ["latents", "text_context"]
    sort!(collect(keys(header.tensors))) == expected_tensors ||
        push!(errors, "cache tensor set does not match its format")
    if haskey(header.tensors, "latents")
        length(header.tensors["latents"].shape) == 4 ||
            push!(errors, "latent tensor rank must be 4")
    end
    if haskey(header.tensors, "text_context")
        length(header.tensors["text_context"].shape) == 2 ||
            push!(errors, "text context tensor rank must be 2")
    end
    if format == "reels-wan-i2v-preprocess-cache-v1"
        if haskey(header.tensors, "conditioning_video")
            length(header.tensors["conditioning_video"].shape) == 4 ||
                push!(errors, "conditioning video tensor rank must be 4")
            haskey(header.tensors, "latents") &&
                header.tensors["conditioning_video"].shape[2:end] !=
                    header.tensors["latents"].shape[2:end] &&
                push!(errors,
                    "conditioning and latent dimensions differ")
        end
        if haskey(header.tensors, "image_features")
            image_shape = header.tensors["image_features"].shape
            length(image_shape) == 2 ||
                push!(errors, "image feature tensor rank must be 2")
            length(image_shape) == 2 && image_shape[1] != 1280 &&
                push!(errors,
                    "image features must have 1280 channels")
        end
    end
    (valid=isempty(errors), errors=errors, header=header)
end

function load_preprocess_cache(path::AbstractString; expected_key=nothing)
    inspected = inspect_preprocess_cache(path; expected_key=expected_key)
    inspected.valid ||
        throw(ArgumentError(join(inspected.errors, "; ")))
    header = inspected.header
    latent_shape = header.tensors["latents"].shape
    text_shape = header.tensors["text_context"].shape
    source = open_tensor_source(path)
    latents = load_state_tensor(source,
        TensorSpec("latents", latent_shape, ROW_MAJOR_SOURCE))
    text_context = load_state_tensor(source,
        TensorSpec("text_context", text_shape, ROW_MAJOR_SOURCE))
    i2v = get(header.metadata, "format", "") ==
        "reels-wan-i2v-preprocess-cache-v1"
    conditioning_video = i2v ? load_state_tensor(source,
        TensorSpec("conditioning_video",
            header.tensors["conditioning_video"].shape,
            ROW_MAJOR_SOURCE)) : nothing
    image_features = i2v ? load_state_tensor(source,
        TensorSpec("image_features",
            header.tensors["image_features"].shape,
            ROW_MAJOR_SOURCE)) : nothing
    (latents=latents, text_context=text_context,
     conditioning_video=conditioning_video,
     image_features=image_features,
     metadata=header.metadata)
end

cache_is_valid(path::AbstractString, key::AbstractString) =
    isfile(path) && inspect_preprocess_cache(path; expected_key=key).valid

function write_ltx23_preprocess_cache(path::AbstractString,
                                      key::AbstractString,
                                      latents::AbstractMatrix,
                                      text_context::AbstractMatrix,
                                      positions::AbstractMatrix;
                                      metadata=Dict{String,String}())
    size(positions, 1) == 3 ||
        throw(DimensionMismatch("LTX positions must be (3,tokens)"))
    size(latents, 2) == size(positions, 2) ||
        throw(DimensionMismatch("LTX latent and position token counts differ"))
    info = Dict{String,String}(
        "format" => "reels-ltx23-preprocess-cache-v1",
        "cache_key" => String(key),
        "model_family" => "ltx23")
    merge!(info, metadata)
    write_safetensors(path, Dict(
        "latents" => Array(latents),
        "text_context" => Array(text_context),
        "positions" => Array(positions)); metadata=info)
end

function inspect_ltx23_preprocess_cache(path::AbstractString;
                                        expected_key=nothing)
    header = inspect_safetensors(path)
    errors = String[]
    get(header.metadata, "format", "") ==
        "reels-ltx23-preprocess-cache-v1" ||
        push!(errors, "unsupported LTX cache format")
    expected_key === nothing ||
        get(header.metadata, "cache_key", "") == expected_key ||
        push!(errors, "cache key mismatch")
    sort!(collect(keys(header.tensors))) ==
        ["latents", "positions", "text_context"] ||
        push!(errors, "LTX cache must contain latents, positions, and text_context")
    for name in ("latents", "positions", "text_context")
        haskey(header.tensors, name) &&
            length(header.tensors[name].shape) != 2 &&
            push!(errors, "$name tensor rank must be 2")
    end
    if haskey(header.tensors, "latents") &&
       haskey(header.tensors, "positions")
        header.tensors["positions"].shape[1] == 3 ||
            push!(errors, "positions first dimension must be 3")
        header.tensors["latents"].shape[2] ==
            header.tensors["positions"].shape[2] ||
            push!(errors, "latent and position token counts differ")
    end
    (valid=isempty(errors), errors=errors, header=header)
end

function load_ltx23_preprocess_cache(path::AbstractString;
                                     expected_key=nothing)
    inspected = inspect_ltx23_preprocess_cache(
        path; expected_key=expected_key)
    inspected.valid ||
        throw(ArgumentError(join(inspected.errors, "; ")))
    source = open_tensor_source(path)
    load(name) = load_state_tensor(source, TensorSpec(name,
        inspected.header.tensors[name].shape, ROW_MAJOR_SOURCE))
    (latents=load("latents"), text_context=load("text_context"),
     positions=load("positions"), metadata=inspected.header.metadata)
end

ltx23_cache_is_valid(path::AbstractString, key::AbstractString) =
    isfile(path) &&
    inspect_ltx23_preprocess_cache(path; expected_key=key).valid

"""
Patch one native LTX VAE latent volume, encode its caption with Gemma and the
LTX connector, and atomically persist the complete training cache entry.
"""
function write_ltx23_preprocess_cache(cache_dir::AbstractString,
                                      backend::LTX23,
                                      conditioner::LTXTextConditioner,
                                      sample::VideoSample,
                                      bucket::BucketAssignment,
                                      identity::PreprocessIdentity,
                                      latent_volume::AbstractArray{<:Real,4})
    key = preprocess_cache_key(sample, bucket, identity)
    path = cache_entry_path(cache_dir, key)
    ltx23_cache_is_valid(path, key) &&
        return (path=path, key=key, created=false)

    encoded = encode_text(backend, conditioner, sample.caption)
    write_ltx23_preprocess_cache(cache_dir, sample, bucket, identity,
                                 latent_volume, encoded)
end

function write_ltx23_preprocess_cache(cache_dir::AbstractString,
                                      sample::VideoSample,
                                      bucket::BucketAssignment,
                                      identity::PreprocessIdentity,
                                      latent_volume::AbstractArray{<:Real,4},
                                      encoded::LTXTextEncoding)
    key = preprocess_cache_key(sample, bucket, identity)
    path = cache_entry_path(cache_dir, key)
    ltx23_cache_is_valid(path, key) &&
        return (path=path, key=key, created=false)
    batched = reshape(latent_volume, size(latent_volume)..., 1)
    patched = ltx23_patchify_latents(batched)
    positions = ltx23_patch_positions(
        size(latent_volume, 2), size(latent_volume, 3),
        size(latent_volume, 4); fps=bucket.fps)
    latents = dropdims(patched; dims=3)
    context = dropdims(encoded.context; dims=3)
    positions = dropdims(positions; dims=3)
    latents isa Array || (latents = Array(latents))
    context isa Array || (context = Array(context))
    positions isa Array || (positions = Array(positions))
    metadata = Dict{String,String}(
        "caption_token_length" => string(only(encoded.lengths)),
        "text_tokens" => string(size(context, 2)),
        "text_features" => string(size(context, 1)),
        "latent_frames" => string(size(latent_volume, 2)),
        "latent_height" => string(size(latent_volume, 3)),
        "latent_width" => string(size(latent_volume, 4)),
        "fps" => string(bucket.fps),
    )
    write_ltx23_preprocess_cache(path, key, latents, context, positions;
                                 metadata=metadata)
    (path=path, key=key, created=true)
end

function build_ltx23_preprocess_cache(cache_dir::AbstractString,
                                      backend::LTX23,
                                      conditioner::LTXTextConditioner,
                                      vae::LTXVideoVAEEncoder,
                                      sample::VideoSample,
                                      identity::PreprocessIdentity,
                                      resolution_buckets,
                                      frame_buckets;
                                      target_fps,
                                      temporal_crop=:center,
                                      ffmpeg=ffmpeg_executable(),
                                      ffprobe=ffprobe_executable())
    decoded = decode_video_sample(sample, resolution_buckets, frame_buckets;
        target_fps=target_fps, temporal_crop=temporal_crop,
        normalization=:minus_one_one, ffmpeg=ffmpeg, ffprobe=ffprobe)
    key = preprocess_cache_key(sample, decoded.bucket, identity)
    path = cache_entry_path(cache_dir, key)
    ltx23_cache_is_valid(path, key) &&
        return (path=path, key=key, created=false, bucket=decoded.bucket)
    latent_volume = encode_video(backend, vae, decoded.frames)
    written = write_ltx23_preprocess_cache(cache_dir, backend, conditioner,
        sample, decoded.bucket, identity, latent_volume)
    merge(written, (bucket=decoded.bucket,))
end

"""
Encode a sample caption with native UMT5 and atomically write it beside an
already-computed Wan latent. This is the text-conditioning half of Wan cache
construction; VAE preprocessing supplies `latents`.
"""
function write_wan_preprocess_cache(cache_dir::AbstractString,
                                    backend::Wan21,
                                    conditioner::WanTextConditioner,
                                    sample::VideoSample,
                                    bucket::BucketAssignment,
                                    identity::PreprocessIdentity,
                                    latents::AbstractArray;
                                    conditioning_video=nothing,
                                    image_features=nothing)
    key = preprocess_cache_key(sample, bucket, identity)
    path = cache_entry_path(cache_dir, key)
    if cache_is_valid(path, key)
        return (path=path, key=key, created=false)
    end
    encoded = encode_text(backend, conditioner, sample.caption)
    context = dropdims(encoded.context; dims=3)
    context isa Array || (context = Array(context))
    metadata = Dict{String,String}(
        "caption_token_length" => string(only(encoded.lengths)),
        "text_tokens" => string(size(context, 2)),
        "text_features" => string(size(context, 1)),
    )
    write_preprocess_cache(path, key, latents, context; metadata=metadata,
        conditioning_video=conditioning_video,
        image_features=image_features)
    (path=path, key=key, created=true)
end

function build_wan_preprocess_cache(cache_dir::AbstractString,
                                    backend::Wan21,
                                    conditioner::WanTextConditioner,
                                    vae::Union{WanVAEEncoder,WanVAE},
                                    sample::VideoSample,
                                    identity::PreprocessIdentity,
                                    resolution_buckets,
                                    frame_buckets;
                                    target_fps,
                                    temporal_crop=:center,
                                    image_encoder=nothing,
                                    ffmpeg=ffmpeg_executable(),
                                    ffprobe=ffprobe_executable())
    decoded = decode_video_sample(sample, resolution_buckets, frame_buckets;
        target_fps=target_fps, temporal_crop=temporal_crop,
        normalization=:minus_one_one, ffmpeg=ffmpeg, ffprobe=ffprobe)
    key = preprocess_cache_key(sample, decoded.bucket, identity)
    path = cache_entry_path(cache_dir, key)
    cache_is_valid(path, key) &&
        return (path=path, key=key, created=false, bucket=decoded.bucket)
    stage = _wan_latent_stage_path(path)
    latents = _load_wan_latent_stage(stage, key)
    if latents === nothing
        latents = encode_video(backend, vae, decoded.frames)
        latents isa Array || (latents = Array(latents))
        _write_wan_latent_stage(stage, key, latents)
    end
    conditioning_video = nothing
    image_features = nothing
    if wan21_config(backend.variant).model_type === :i2v
        image_encoder === nothing &&
            throw(ArgumentError(
                "Wan I2V preprocessing requires a native image_encoder"))
        first_frame = @view decoded.frames[:, 1, :, :]
        image_features = encode_image(backend, image_encoder, first_frame)
        image_features isa Array || (image_features = Array(image_features))
        prepared = prepare_wan_i2v_conditioning(
            backend, vae, decoded.frames, image_features;
            target_latents=latents)
        conditioning_video = prepared.conditioning_video
        conditioning_video isa Array ||
            (conditioning_video = Array(conditioning_video))
    end
    written = write_wan_preprocess_cache(cache_dir, backend, conditioner,
        sample, decoded.bucket, identity, latents;
        conditioning_video=conditioning_video,
        image_features=image_features)
    isfile(written.path) && rm(stage; force=true)
    merge(written, (bucket=decoded.bucket,))
end

function _validation_cache_key(family::AbstractString,
                               prompt::AbstractString,
                               shape, identity::PreprocessIdentity,
                               seed::Integer)
    bytes2hex(sha256(json_encode(Dict(
        "format" => "reels-validation-cache-v1",
        "model_family" => String(family),
        "prompt" => String(prompt),
        "latent_shape" => collect(shape),
        "model_checkpoint" => identity.model_checkpoint,
        "text_encoder_checkpoint" => identity.text_encoder_checkpoint,
        "image_encoder_checkpoint" => identity.image_encoder_checkpoint,
        "vae_checkpoint" => identity.vae_checkpoint,
        "preprocessing_version" => identity.preprocessing_version,
        "vae_scale" => identity.vae_scale,
        "dtype" => identity.dtype,
        "seed" => Int(seed),
    ))))
end

function _write_validation_cache(cache_dir::AbstractString,
                                 key::AbstractString,
                                 family::AbstractString,
                                 prompt::AbstractString,
                                 tensors::AbstractDict;
                                 seed::Integer)
    path = joinpath(cache_dir, "validation", "$key.safetensors")
    if isfile(path)
        inspected = inspect_validation_cache(path)
        inspected.valid &&
            get(inspected.header.metadata, "cache_key", "") == key &&
            return (path=path, key=String(key), created=false)
    end
    write_safetensors(path, tensors; metadata=Dict(
        "format" => "reels-validation-cache-v1",
        "cache_key" => String(key),
        "model_family" => String(family),
        "prompt" => String(prompt),
        "seed" => string(seed),
    ))
    (path=path, key=String(key), created=true)
end

function inspect_validation_cache(path::AbstractString)
    header = inspect_safetensors(path)
    errors = String[]
    get(header.metadata, "format", "") == "reels-validation-cache-v1" ||
        push!(errors, "unsupported validation cache format")
    family = get(header.metadata, "model_family", "")
    actual = sort!(collect(keys(header.tensors)))
    expected = family == "wan21" ?
        (actual == ["conditioning_video", "image_features", "noise",
                    "text_context"] ? actual : ["noise", "text_context"]) :
        family == "ltx23" ?
            ["noise", "positions", "text_context"] : String[]
    isempty(expected) && push!(errors, "unsupported validation model family")
    actual == expected ||
        push!(errors, "validation cache tensor inventory is invalid")
    if haskey(header.tensors, "noise")
        expected_rank = family == "wan21" ? 5 : 3
        length(header.tensors["noise"].shape) == expected_rank ||
            push!(errors, "validation noise rank is invalid")
    end
    haskey(header.tensors, "text_context") &&
        length(header.tensors["text_context"].shape) != 3 &&
        push!(errors, "validation text context must be rank 3")
    if family == "wan21" && haskey(header.tensors, "conditioning_video")
        length(header.tensors["conditioning_video"].shape) == 5 ||
            push!(errors, "I2V validation conditioning must be rank 5")
        length(header.tensors["image_features"].shape) == 3 ||
            push!(errors, "I2V validation image features must be rank 3")
        header.tensors["conditioning_video"].shape[1] == 20 ||
            push!(errors,
                "I2V validation conditioning must have 20 channels")
        header.tensors["image_features"].shape[1] == 1280 ||
            push!(errors,
                "I2V validation image features must have 1280 channels")
    end
    if family == "ltx23" && haskey(header.tensors, "positions")
        header.tensors["positions"].shape[1] == 3 ||
            push!(errors, "LTX validation positions must start with dimension 3")
        if haskey(header.tensors, "noise")
            header.tensors["positions"].shape[2:end] ==
                header.tensors["noise"].shape[2:end] ||
                push!(errors, "LTX validation positions and noise tokens differ")
        end
    end
    isempty(get(header.metadata, "prompt", "")) &&
        push!(errors, "validation prompt is missing")
    (valid=isempty(errors), errors=errors, header=header)
end

function _validation_prompt_rng(seed::Integer, prompt::AbstractString)
    digest = sha256(String(prompt))
    prompt_seed = zero(UInt64)
    for index in 1:8
        prompt_seed |= UInt64(digest[index]) << (8 * (index - 1))
    end
    Xoshiro(UInt64(seed) ⊻ prompt_seed)
end

function load_validation_cache(path::AbstractString;
                               device=:cpu, precision=:fp32)
    inspected = inspect_validation_cache(path)
    inspected.valid ||
        throw(ArgumentError(join(inspected.errors, "; ")))
    source = open_tensor_source(path)
    load(name) = load_state_tensor(source, TensorSpec(name,
        inspected.header.tensors[name].shape, ROW_MAJOR_SOURCE))
    transfer = array_transfer(device, precision)
    prompt = inspected.header.metadata["prompt"]
    family = inspected.header.metadata["model_family"]
    noise = transfer(load("noise"))
    context = transfer(load("text_context"))
    if family == "wan21"
        i2v = haskey(inspected.header.tensors, "conditioning_video")
        conditioning = i2v ? transfer(load("conditioning_video")) : nothing
        image = i2v ? transfer(load("image_features")) : nothing
        return WanValidationBatch(prompt, noise, context;
            conditioning_video=conditioning, image_features=image)
    end
    positions = load("positions")
    LTXValidationBatch(prompt, noise, context, positions)
end

function build_wan_validation_caches(cache_dir::AbstractString,
                                     backend::Wan21,
                                     conditioner::WanTextConditioner,
                                     template_entry::AbstractString,
                                     prompts::AbstractVector{<:AbstractString},
                                     identity::PreprocessIdentity;
                                     seed::Integer)
    template = load_preprocess_cache(template_entry)
    map(prompts) do prompt
        rng = _validation_prompt_rng(seed, prompt)
        encoded = encode_text(backend, conditioner, prompt)
        context = encoded.context isa Array ?
            encoded.context : Array(encoded.context)
        noise = randn(rng, Float32, size(template.latents)..., 1)
        key = _validation_cache_key(
            "wan21", prompt, size(noise), identity, seed)
        tensors = Dict{String,AbstractArray}(
            "noise" => noise, "text_context" => context)
        if template.conditioning_video !== nothing
            tensors["conditioning_video"] = reshape(
                template.conditioning_video,
                size(template.conditioning_video)..., 1)
            tensors["image_features"] = reshape(template.image_features,
                size(template.image_features)..., 1)
        end
        _write_validation_cache(cache_dir, key, "wan21", prompt,
            tensors; seed=seed)
    end
end

function build_ltx23_validation_caches(cache_dir::AbstractString,
                                       backend::LTX23,
                                       conditioner::LTXTextConditioner,
                                       template_entry::AbstractString,
                                       prompts::AbstractVector{<:AbstractString},
                                       identity::PreprocessIdentity;
                                       seed::Integer)
    template = load_ltx23_preprocess_cache(template_entry)
    positions = reshape(template.positions, size(template.positions)..., 1)
    map(prompts) do prompt
        rng = _validation_prompt_rng(seed, prompt)
        encoded = encode_text(backend, conditioner, prompt)
        context = encoded.context isa Array ?
            encoded.context : Array(encoded.context)
        noise = randn(rng, Float32, size(template.latents)..., 1)
        key = _validation_cache_key(
            "ltx23", prompt, size(noise), identity, seed)
        _write_validation_cache(cache_dir, key, "ltx23", prompt,
            Dict("noise" => noise, "text_context" => context,
                 "positions" => positions); seed=seed)
    end
end
