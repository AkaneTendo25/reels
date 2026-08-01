using Libdl

const _SENTENCEPIECE_LIBRARY = Ref{Union{Nothing,String}}(nothing)

function sentencepiece_library()
    configured = get(ENV, "REELS_SENTENCEPIECE_LIBRARY", "")
    candidates = isempty(configured) ? String[
        joinpath(pkgdir(Reels), "deps", "usr", "lib",
            "libreels_sentencepiece." * Libdl.dlext),
    ] : String[configured]
    for candidate in candidates
        isfile(candidate) && return candidate
    end
    throw(ArgumentError(
        "native SentencePiece bridge is unavailable; run `julia --project=. deps/build.jl` " *
        "or set REELS_SENTENCEPIECE_LIBRARY"))
end

_sp_symbol(library_handle::Ptr{Cvoid}, name::Symbol) =
    Libdl.dlsym(library_handle, name)

function _sp_error(library_handle::Ptr{Cvoid}, operation::String)
    pointer = ccall(_sp_symbol(library_handle, :reels_sp_last_error), Cstring, ())
    detail = pointer == C_NULL ? "unknown native error" : unsafe_string(pointer)
    ErrorException("$operation failed: $detail")
end

mutable struct SentencePieceTokenizer
    handle::Ptr{Cvoid}
    library_handle::Ptr{Cvoid}
    library::String
    model_path::String
    vocab_size::Int
    pad_id::Int32
    eos_id::Int32
    unk_id::Int32
end

function Base.close(tokenizer::SentencePieceTokenizer)
    if tokenizer.handle != C_NULL
        ccall(_sp_symbol(tokenizer.library_handle, :reels_sp_destroy),
            Cvoid, (Ptr{Cvoid},),
            tokenizer.handle)
        tokenizer.handle = C_NULL
    end
    nothing
end

function SentencePieceTokenizer(model_path::AbstractString;
                                library::AbstractString=sentencepiece_library())
    path = abspath(String(model_path))
    isfile(path) || throw(ArgumentError("SentencePiece model does not exist: $path"))
    lib = abspath(String(library))
    library_handle = Libdl.dlopen(lib)
    handle = ccall(_sp_symbol(library_handle, :reels_sp_create), Ptr{Cvoid}, ())
    handle == C_NULL &&
        throw(_sp_error(library_handle, "SentencePiece construction"))
    status = ccall(_sp_symbol(library_handle, :reels_sp_load),
        Cint, (Ptr{Cvoid}, Cstring),
        handle, path)
    if status != 0
        ccall(_sp_symbol(library_handle, :reels_sp_destroy),
            Cvoid, (Ptr{Cvoid},), handle)
        throw(_sp_error(library_handle, "loading SentencePiece model"))
    end
    vocab = ccall(_sp_symbol(library_handle, :reels_sp_piece_size),
        Cint, (Ptr{Cvoid},), handle)
    pad = ccall(_sp_symbol(library_handle, :reels_sp_pad_id),
        Cint, (Ptr{Cvoid},), handle)
    eos = ccall(_sp_symbol(library_handle, :reels_sp_eos_id),
        Cint, (Ptr{Cvoid},), handle)
    unk = ccall(_sp_symbol(library_handle, :reels_sp_unk_id),
        Cint, (Ptr{Cvoid},), handle)
    tokenizer = SentencePieceTokenizer(handle, library_handle, lib, path, Int(vocab),
        Int32(pad), Int32(eos), Int32(unk))
    finalizer(close, tokenizer)
    tokenizer
end

function clean_wan_caption(text::AbstractString)
    value = replace(String(text),
        "&quot;" => "\"", "&#34;" => "\"", "&apos;" => "'",
        "&#39;" => "'", "&amp;" => "&", "&lt;" => "<", "&gt;" => ">",
        "&nbsp;" => " ")
    strip(replace(value, r"\s+" => " "))
end

function sentencepiece_ids(tokenizer::SentencePieceTokenizer, text::AbstractString)
    tokenizer.handle == C_NULL && throw(ArgumentError("tokenizer is closed"))
    cleaned = clean_wan_caption(text)
    length_ref = Ref{Csize_t}(0)
    status = ccall(_sp_symbol(tokenizer.library_handle, :reels_sp_encode), Cint,
        (Ptr{Cvoid}, Cstring, Ptr{Int32}, Csize_t, Ref{Csize_t}),
        tokenizer.handle, cleaned, C_NULL, 0, length_ref)
    status == 0 ||
        throw(_sp_error(tokenizer.library_handle, "SentencePiece encode"))
    ids = Vector{Int32}(undef, Int(length_ref[]))
    isempty(ids) && return ids
    status = ccall(_sp_symbol(tokenizer.library_handle, :reels_sp_encode), Cint,
        (Ptr{Cvoid}, Cstring, Ptr{Int32}, Csize_t, Ref{Csize_t}),
        tokenizer.handle, cleaned, ids, length(ids), length_ref)
    status == 0 ||
        throw(_sp_error(tokenizer.library_handle, "SentencePiece encode"))
    ids
end

struct TokenizedText
    ids::Matrix{Int32}
    mask::BitMatrix
    lengths::Vector{Int}
end

function tokenize_wan(tokenizer::SentencePieceTokenizer,
                      texts::AbstractVector{<:AbstractString};
                      max_length::Integer=512)
    max_length > 0 || throw(ArgumentError("max_length must be positive"))
    tokenizer.pad_id >= 0 || throw(ArgumentError("tokenizer has no padding id"))
    tokenizer.eos_id >= 0 || throw(ArgumentError("tokenizer has no EOS id"))
    batch = length(texts)
    ids = fill(tokenizer.pad_id, Int(max_length), batch)
    mask = falses(Int(max_length), batch)
    lengths = Vector{Int}(undef, batch)
    for (column, text) in enumerate(texts)
        encoded = sentencepiece_ids(tokenizer, text)
        keep = min(length(encoded), Int(max_length) - 1)
        keep > 0 && copyto!(view(ids, 1:keep, column), view(encoded, 1:keep))
        ids[keep + 1, column] = tokenizer.eos_id
        mask[1:keep + 1, column] .= true
        lengths[column] = keep + 1
    end
    TokenizedText(ids, mask, lengths)
end

tokenize_wan(tokenizer::SentencePieceTokenizer, text::AbstractString; kwargs...) =
    tokenize_wan(tokenizer, [text]; kwargs...)

"""
Tokenize Gemma prompts with the Hugging Face Gemma convention used by LTX:
prepend BOS, do not append EOS, truncate on the right, and pad on the left.
SentencePiece and model token IDs are both zero-based.
"""
function tokenize_gemma(tokenizer::SentencePieceTokenizer,
                        texts::AbstractVector{<:AbstractString};
                        max_length::Integer=1024, bos_id::Integer=2)
    max_length > 0 || throw(ArgumentError("max_length must be positive"))
    0 <= bos_id < tokenizer.vocab_size ||
        throw(ArgumentError("Gemma BOS ID is outside the tokenizer vocabulary"))
    padding_id = tokenizer.pad_id >= 0 ? tokenizer.pad_id : tokenizer.eos_id
    padding_id >= 0 ||
        throw(ArgumentError("Gemma tokenizer has neither padding nor EOS ID"))
    batch = length(texts)
    ids = fill(Int32(padding_id), Int(max_length), batch)
    mask = falses(Int(max_length), batch)
    lengths = Vector{Int}(undef, batch)
    for (column, text) in enumerate(texts)
        pieces = sentencepiece_ids(tokenizer, strip(String(text)))
        keep = min(length(pieces), Int(max_length) - 1)
        token_count = keep + 1
        first_token = Int(max_length) - token_count + 1
        ids[first_token, column] = Int32(bos_id)
        keep > 0 && copyto!(
            view(ids, first_token + 1:Int(max_length), column),
            view(pieces, 1:keep))
        mask[first_token:Int(max_length), column] .= true
        lengths[column] = token_count
    end
    TokenizedText(ids, mask, lengths)
end

tokenize_gemma(tokenizer::SentencePieceTokenizer,
               text::AbstractString; kwargs...) =
    tokenize_gemma(tokenizer, [text]; kwargs...)
