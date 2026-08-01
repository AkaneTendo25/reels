struct TensorInfo
    dtype::String
    shape::Vector{Int}
    data_offsets::Tuple{Int,Int}
end

struct SafeTensorHeader
    metadata::Dict{String,String}
    tensors::Dict{String,TensorInfo}
    data_start::Int64
end

_read_vector(io, ::Type{T}, n::Integer) where T =
    read!(io, Vector{T}(undef, Int(n)))

function inspect_safetensors(path::AbstractString)
    open(path, "r") do io
        filesize(path) >= 8 || throw(ArgumentError("truncated SafeTensors file"))
        header_len = read(io, UInt64)
        header_len <= 100_000_000 ||
            throw(ArgumentError("unreasonable SafeTensors header length $header_len"))
        8 + header_len <= filesize(path) ||
            throw(ArgumentError("SafeTensors header exceeds file size"))
        raw = parse_json(String(_read_vector(io, UInt8, header_len)))
        metadata = Dict{String,String}()
        tensors = Dict{String,TensorInfo}()
        for (name, value) in pairs(raw)
            if name == "__metadata__"
                foreach(pairs(value)) do (k, v)
                    metadata[k] = String(v)
                end
            else
                offsets = Int.(value["data_offsets"])
                length(offsets) == 2 && 0 <= offsets[1] <= offsets[2] ||
                    throw(ArgumentError("invalid offsets for tensor $name"))
                tensors[name] = TensorInfo(String(value["dtype"]), Int.(value["shape"]),
                    (offsets[1], offsets[2]))
            end
        end
        ranges = sort([(v.data_offsets..., k) for (k, v) in tensors])
        previous = 0
        for (lo, hi, name) in ranges
            lo >= previous || throw(ArgumentError("overlapping tensor data at $name"))
            8 + Int(header_len) + hi <= filesize(path) ||
                throw(ArgumentError("tensor $name exceeds file size"))
            previous = hi
        end
        SafeTensorHeader(metadata, tensors, Int64(8 + header_len))
    end
end

const SAFETENSOR_DTYPES = Dict(
    "F32" => (Float32, 4), "F64" => (Float64, 8),
    "F16" => (Float16, 2), "BF16" => (BFloat16, 2),
    "I64" => (Int64, 8), "I32" => (Int32, 4),
    "I16" => (Int16, 2), "I8" => (Int8, 1),
    "U8" => (UInt8, 1), "BOOL" => (Bool, 1),
)

function load_safetensor(path::AbstractString, name::AbstractString)
    header = inspect_safetensors(path)
    info = get(header.tensors, name, nothing)
    info === nothing && throw(KeyError(name))
    spec = get(SAFETENSOR_DTYPES, info.dtype, nothing)
    spec === nothing && throw(ArgumentError("unsupported SafeTensors dtype $(info.dtype)"))
    T, bytes = spec
    count = prod(info.shape; init=1)
    info.data_offsets[2] - info.data_offsets[1] == count * bytes ||
        throw(ArgumentError("byte length does not match shape for tensor $name"))
    open(path, "r") do io
        seek(io, header.data_start + info.data_offsets[1])
        # SafeTensors uses row-major dimensions. Reverse dimensions creates a
        # zero-copy Julia column-major view with equivalent logical indexing.
        reshape(_read_vector(io, T, count), reverse(info.shape)...)
    end
end

_dtype(::Type{Float32}) = "F32"
_dtype(::Type{Float64}) = "F64"
_dtype(::Type{Float16}) = "F16"
_dtype(::Type{BFloat16}) = "BF16"
_dtype(::Type{Int64}) = "I64"
_dtype(::Type{Int32}) = "I32"
_dtype(::Type{Int16}) = "I16"
_dtype(::Type{Int8}) = "I8"
_dtype(::Type{UInt8}) = "U8"
_dtype(::Type{Bool}) = "BOOL"

function _bf16_to_float32(array::AbstractArray{BFloat16})
    words = UInt32.(reinterpret(UInt16, vec(array))) .<< 16
    reshape(reinterpret(Float32, words), size(array))
end

function _copy_model_tensor!(destination::AbstractArray,
                             source::AbstractArray)
    values = eltype(destination) === Float32 &&
             eltype(source) === BFloat16 ?
        _bf16_to_float32(source) : source
    copyto!(destination, values)
end

function write_safetensors(path::AbstractString, tensors::AbstractDict;
                           metadata=Dict{String,String}())
    names = sort!(String.(collect(keys(tensors))))
    header = Dict{String,Any}()
    isempty(metadata) || (header["__metadata__"] = Dict(sort!(collect(metadata))))
    offset = 0
    for name in names
        a = tensors[name]
        nbytes = sizeof(eltype(a)) * length(a)
        header[name] = Dict("dtype" => _dtype(eltype(a)),
            "shape" => collect(size(a)),
            "data_offsets" => [offset, offset + nbytes])
        offset += nbytes
    end
    encoded = json_encode(header)
    padding = mod(-ncodeunits(encoded), 8)
    encoded *= " "^padding
    isempty(dirname(path)) || mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, UInt64(ncodeunits(encoded)))
        write(io, codeunits(encoded))
        for name in names
            tensor = tensors[name]
            row_major = ndims(tensor) <= 1 ? tensor :
                permutedims(tensor, reverse(1:ndims(tensor)))
            write(io, vec(row_major))
        end
    end
    mv(tmp, path; force=true)
    path
end

abstract type AbstractTensorSource end

struct SingleTensorSource <: AbstractTensorSource
    path::String
    header::SafeTensorHeader
end

struct ShardedTensorSource <: AbstractTensorSource
    index_path::String
    key_to_path::Dict{String,String}
    headers::Dict{String,SafeTensorHeader}
end

struct KeyMappedTensorSource{S<:AbstractTensorSource} <: AbstractTensorSource
    source::S
    key_to_source_key::Dict{String,String}
end

tensor_keys(source::SingleTensorSource) = collect(keys(source.header.tensors))
tensor_keys(source::ShardedTensorSource) = collect(keys(source.key_to_path))
tensor_keys(source::KeyMappedTensorSource) =
    collect(keys(source.key_to_source_key))

function _safe_shard_path(index_path, shard_name)
    isabspath(shard_name) &&
        throw(ArgumentError("absolute shard path is not allowed: $shard_name"))
    base = abspath(dirname(index_path))
    resolved = abspath(joinpath(base, shard_name))
    separator = Sys.iswindows() ? '\\' : '/'
    (resolved == base || startswith(resolved, base * separator)) ||
        throw(ArgumentError("shard escapes checkpoint directory: $shard_name"))
    isfile(resolved) || throw(ArgumentError("missing SafeTensors shard: $resolved"))
    resolved
end

function open_tensor_source(path::AbstractString)
    if isdir(path)
        entries = readdir(path; join=true)
        indexes = sort!(filter(
            entry -> isfile(entry) &&
                endswith(lowercase(entry), ".safetensors.index.json"),
            entries))
        if length(indexes) == 1
            return open_tensor_source(only(indexes))
        elseif length(indexes) > 1
            throw(ArgumentError(
                "checkpoint directory contains multiple SafeTensors indexes: " *
                join(basename.(indexes), ", ")))
        end
        tensors = sort!(filter(
            entry -> isfile(entry) &&
                endswith(lowercase(entry), ".safetensors"),
            entries))
        length(tensors) == 1 && return open_tensor_source(only(tensors))
        isempty(tensors) &&
            throw(ArgumentError(
                "checkpoint directory contains no SafeTensors source: $path"))
        throw(ArgumentError(
            "checkpoint directory contains multiple SafeTensors files " *
            "without an index: " * join(basename.(tensors), ", ")))
    end
    if endswith(lowercase(path), ".index.json")
        raw = parse_json(read(path, String))
        haskey(raw, "weight_map") ||
            throw(ArgumentError("SafeTensors index lacks weight_map"))
        weight_map = raw["weight_map"]
        weight_map isa AbstractDict ||
            throw(ArgumentError("SafeTensors weight_map must be an object"))
        key_to_path = Dict{String,String}()
        for (key, shard) in weight_map
            shard isa AbstractString ||
                throw(ArgumentError("shard name for $key must be a string"))
            key_to_path[key] = _safe_shard_path(path, shard)
        end
        headers = Dict(file => inspect_safetensors(file)
                       for file in unique(values(key_to_path)))
        seen = Dict{String,String}()
        for (file, header) in headers, key in keys(header.tensors)
            haskey(seen, key) &&
                throw(ArgumentError("tensor $key occurs in multiple shards"))
            seen[key] = file
        end
        for (key, file) in key_to_path
            haskey(headers[file].tensors, key) ||
                throw(ArgumentError("index maps $key to a shard that does not contain it"))
        end
        unmapped = sort!(setdiff(collect(keys(seen)), collect(keys(key_to_path))))
        isempty(unmapped) ||
            throw(ArgumentError("shards contain tensors absent from index: " *
                                join(unmapped, ", ")))
        return ShardedTensorSource(abspath(path), key_to_path, headers)
    end
    SingleTensorSource(abspath(path), inspect_safetensors(path))
end

function load_safetensor(source::SingleTensorSource, name::AbstractString)
    load_safetensor(source.path, name)
end
function load_safetensor(source::ShardedTensorSource, name::AbstractString)
    file = get(source.key_to_path, name, nothing)
    file === nothing && throw(KeyError(name))
    load_safetensor(file, name)
end
function load_safetensor(source::KeyMappedTensorSource, name::AbstractString)
    source_key = get(source.key_to_source_key, name, nothing)
    source_key === nothing && throw(KeyError(name))
    load_safetensor(source.source, source_key)
end

function write_sharded_safetensors(index_path::AbstractString,
                                   shards::AbstractDict; metadata=Dict())
    base = dirname(index_path)
    isempty(base) || mkpath(base)
    weight_map = Dict{String,String}()
    total_size = 0
    for shard_name in sort!(String.(collect(keys(shards))))
        tensors = shards[shard_name]
        for key in keys(tensors)
            haskey(weight_map, String(key)) &&
                throw(ArgumentError("tensor $(String(key)) assigned to multiple shards"))
            weight_map[String(key)] = shard_name
            total_size += sizeof(eltype(tensors[key])) * length(tensors[key])
        end
        write_safetensors(joinpath(base, shard_name), tensors; metadata=metadata)
    end
    index = Dict("metadata" => Dict("total_size" => total_size),
                 "weight_map" => weight_map)
    encoded = json_encode(index)
    tmp = index_path * ".tmp"
    open(tmp, "w") do io
        write(io, encoded)
    end
    mv(tmp, index_path; force=true)
    index_path
end
