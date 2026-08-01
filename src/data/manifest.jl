struct VideoSample
    id::String
    video::String
    caption::String
end

const SAMPLE_KEYS = Set(["video", "caption", "caption_file"])

function _sample_from_table(table, manifest_path, index; caption_prefix="")
    table isa AbstractDict ||
        throw(ArgumentError("sample $index must be an object"))
    unknown = sort!(setdiff(String.(collect(keys(table))), collect(SAMPLE_KEYS)))
    isempty(unknown) ||
        throw(ArgumentError("sample $index has unknown keys: $(join(unknown, ", "))"))
    haskey(table, "video") ||
        throw(ArgumentError("sample $index is missing video"))
    direct = haskey(table, "caption")
    from_file = haskey(table, "caption_file")
    direct ⊻ from_file ||
        throw(ArgumentError("sample $index must set exactly one of caption or caption_file"))
    base = dirname(abspath(manifest_path))
    resolve(path) = normpath(isabspath(path) ? path : joinpath(base, path))
    video = resolve(String(table["video"]))
    isfile(video) ||
        throw(ArgumentError("sample $index video does not exist: $video"))
    caption = if direct
        String(table["caption"])
    else
        caption_path = resolve(String(table["caption_file"]))
        isfile(caption_path) ||
            throw(ArgumentError("sample $index caption file does not exist: $caption_path"))
        read(caption_path, String)
    end
    caption = String(caption_prefix) * strip(caption)
    isempty(caption) &&
        throw(ArgumentError("sample $index caption is empty"))
    id = bytes2hex(sha256(string(video, '\0', caption)))
    VideoSample(id, video, caption)
end

function _jsonl_samples(path)
    tables = Any[]
    for (line_number, line) in enumerate(eachline(path))
        isempty(strip(line)) && continue
        table = try
            parse_json(line)
        catch error
            throw(ArgumentError("invalid JSONL at line $line_number: $(sprint(showerror, error))"))
        end
        push!(tables, table)
    end
    tables
end

"""
    load_video_manifest(path; caption_prefix="")

Load strict TOML `[[samples]]` or JSONL captioned-video manifests. Relative
media and caption paths resolve against the manifest directory.
"""
function load_video_manifest(path::AbstractString; caption_prefix="")
    isfile(path) || throw(ArgumentError("manifest does not exist: $path"))
    extension = lowercase(splitext(path)[2])
    tables = if extension == ".toml"
        raw = TOML.parsefile(path)
        unknown = sort!(setdiff(String.(collect(keys(raw))), ["samples"]))
        isempty(unknown) ||
            throw(ArgumentError("manifest has unknown keys: $(join(unknown, ", "))"))
        get(raw, "samples", Any[])
    elseif extension in (".jsonl", ".ndjson")
        _jsonl_samples(path)
    else
        throw(ArgumentError("manifest must use .toml, .jsonl, or .ndjson"))
    end
    isempty(tables) && throw(ArgumentError("manifest contains no samples"))
    samples = [_sample_from_table(table, path, index;
        caption_prefix=caption_prefix) for (index, table) in enumerate(tables)]
    ids = [sample.id for sample in samples]
    length(unique(ids)) == length(ids) ||
        throw(ArgumentError("manifest contains duplicate video/caption samples"))
    samples
end
