struct TensorSpec
    source_key::String
    destination_key::String
    source_shape::Vector{Int}
    layout::TensorLayout
    required::Bool
end

TensorSpec(key, shape, layout; destination=key, required=true) =
    TensorSpec(String(key), String(destination), Int.(shape), layout, required)

struct StateDictAudit
    missing::Vector{String}
    unexpected::Vector{String}
    shape_mismatches::Vector{String}
end
Base.isempty(a::StateDictAudit) =
    isempty(a.missing) && isempty(a.unexpected) && isempty(a.shape_mismatches)

function audit_state_dict(header::SafeTensorHeader, specs::AbstractVector{TensorSpec};
                          allow_unexpected=false)
    by_key = Dict(spec.source_key => spec for spec in specs)
    missing = sort!([key for (key, spec) in by_key
                     if spec.required && !haskey(header.tensors, key)])
    unexpected = allow_unexpected ? String[] :
        sort!(setdiff(collect(keys(header.tensors)), collect(keys(by_key))))
    mismatches = String[]
    for (key, spec) in by_key
        haskey(header.tensors, key) || continue
        actual = header.tensors[key].shape
        actual == spec.source_shape || push!(mismatches,
            "$key: expected $(Tuple(spec.source_shape)), found $(Tuple(actual))")
    end
    StateDictAudit(missing, unexpected, sort!(mismatches))
end

function _tensor_dictionary(source::SingleTensorSource)
    source.header.tensors
end
function _tensor_dictionary(source::ShardedTensorSource)
    Dict(key => source.headers[file].tensors[key]
         for (key, file) in source.key_to_path)
end
function _tensor_dictionary(source::KeyMappedTensorSource)
    underlying = _tensor_dictionary(source.source)
    Dict(key => underlying[source_key]
         for (key, source_key) in source.key_to_source_key
         if haskey(underlying, source_key))
end

function audit_state_dict(source::AbstractTensorSource,
                          specs::AbstractVector{TensorSpec}; allow_unexpected=false)
    combined = SafeTensorHeader(Dict{String,String}(), _tensor_dictionary(source), 0)
    audit_state_dict(combined, specs; allow_unexpected=allow_unexpected)
end

function load_state_tensor(path::AbstractString, spec::TensorSpec)
    load_state_tensor(open_tensor_source(path), spec)
end

function load_state_tensor(source::AbstractTensorSource, spec::TensorSpec)
    raw = load_safetensor(source, spec.source_key)
    restored = restore_source_layout(raw, spec.layout)
    size(restored) == Tuple(spec.source_shape) ||
        throw(DimensionMismatch("$(spec.source_key) restored to $(size(restored)); " *
                                "expected $(Tuple(spec.source_shape))"))
    restored
end
