mutable struct JSONCursor
    bytes::Vector{UInt8}
    pos::Int
end

_peek(c) = c.pos <= length(c.bytes) ? c.bytes[c.pos] : 0x00
function _space!(c)
    while _peek(c) in (0x20, 0x09, 0x0a, 0x0d)
        c.pos += 1
    end
end
function _expect!(c, byte)
    _peek(c) == byte || throw(ArgumentError("invalid JSON at byte $(c.pos)"))
    c.pos += 1
end

function _json_string!(c)
    _expect!(c, 0x22)
    out = IOBuffer()
    while true
        b = _peek(c)
        b == 0x00 && throw(ArgumentError("unterminated JSON string"))
        c.pos += 1
        b == 0x22 && return String(take!(out))
        if b == 0x5c
            e = _peek(c); c.pos += 1
            e == 0x22 && write(out, UInt8('"'))
            e == 0x5c && write(out, UInt8('\\'))
            e == 0x2f && write(out, UInt8('/'))
            e == 0x62 && write(out, UInt8('\b'))
            e == 0x66 && write(out, UInt8('\f'))
            e == 0x6e && write(out, UInt8('\n'))
            e == 0x72 && write(out, UInt8('\r'))
            e == 0x74 && write(out, UInt8('\t'))
            e == 0x75 && begin
                c.pos + 3 <= length(c.bytes) || throw(ArgumentError("truncated JSON escape"))
                hex = String(c.bytes[c.pos:c.pos+3]); c.pos += 4
                codepoint = parse(UInt32, hex; base=16)
                if 0xd800 <= codepoint <= 0xdbff
                    c.pos + 5 <= length(c.bytes) ||
                        throw(ArgumentError("truncated JSON surrogate pair"))
                    c.bytes[c.pos:c.pos+1] == UInt8[0x5c, 0x75] ||
                        throw(ArgumentError("invalid JSON surrogate pair"))
                    c.pos += 2
                    low = parse(UInt32,
                        String(c.bytes[c.pos:c.pos+3]); base=16)
                    c.pos += 4
                    0xdc00 <= low <= 0xdfff ||
                        throw(ArgumentError("invalid JSON low surrogate"))
                    codepoint = 0x10000 +
                        ((codepoint - 0xd800) << 10) + (low - 0xdc00)
                elseif 0xdc00 <= codepoint <= 0xdfff
                    throw(ArgumentError("unpaired JSON low surrogate"))
                end
                print(out, Char(codepoint))
            end
            e in (0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74, 0x75) ||
                throw(ArgumentError("invalid JSON escape"))
        else
            b < 0x20 && throw(ArgumentError("control byte in JSON string"))
            write(out, b)
        end
    end
end

function _json_literal!(c, literal::AbstractString, value)
    bytes = collect(codeunits(literal))
    last = c.pos + length(bytes) - 1
    last <= length(c.bytes) && c.bytes[c.pos:last] == bytes ||
        throw(ArgumentError("invalid JSON literal at byte $(c.pos)"))
    c.pos = last + 1
    value
end

function _json_value!(c)
    _space!(c)
    b = _peek(c)
    b == 0x22 && return _json_string!(c)
    b == 0x74 && return _json_literal!(c, "true", true)
    b == 0x66 && return _json_literal!(c, "false", false)
    b == 0x6e && return _json_literal!(c, "null", nothing)
    if b == 0x7b
        c.pos += 1; result = Dict{String,Any}(); _space!(c)
        _peek(c) == 0x7d && (c.pos += 1; return result)
        while true
            _space!(c); key = _json_string!(c); _space!(c); _expect!(c, 0x3a)
            haskey(result, key) && throw(ArgumentError("duplicate JSON key '$key'"))
            result[key] = _json_value!(c); _space!(c)
            _peek(c) == 0x7d && (c.pos += 1; return result)
            _expect!(c, 0x2c)
        end
    elseif b == 0x5b
        c.pos += 1; result = Any[]; _space!(c)
        _peek(c) == 0x5d && (c.pos += 1; return result)
        while true
            push!(result, _json_value!(c)); _space!(c)
            _peek(c) == 0x5d && (c.pos += 1; return result)
            _expect!(c, 0x2c)
        end
    end
    start = c.pos
    while _peek(c) in UInt8.('0':'9') || _peek(c) in (0x2d, 0x2b, 0x2e, 0x45, 0x65)
        c.pos += 1
    end
    c.pos == start && throw(ArgumentError("unsupported JSON value at byte $(c.pos)"))
    token = String(c.bytes[start:c.pos-1])
    occursin(r"[.eE]", token) ? parse(Float64, token) : parse(Int64, token)
end

function parse_json(text::AbstractString)
    c = JSONCursor(collect(codeunits(text)), 1)
    value = _json_value!(c); _space!(c)
    c.pos == length(c.bytes) + 1 || throw(ArgumentError("trailing JSON data"))
    value
end

_json_escape(s) = "\"" * replace(String(s), '\\' => "\\\\", '"' => "\\\"",
    '\n' => "\\n", '\r' => "\\r", '\t' => "\\t") * "\""
json_encode(x::AbstractString) = _json_escape(x)
json_encode(x::Bool) = x ? "true" : "false"
json_encode(::Nothing) = "null"
json_encode(x::Integer) = string(x)
function json_encode(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("JSON cannot encode non-finite values"))
    string(x)
end
json_encode(x::AbstractVector) = "[" * join(json_encode.(x), ",") * "]"
json_encode(x::Tuple) = json_encode(collect(x))
function json_encode(x::AbstractDict)
    keys_sorted = sort!(String.(collect(keys(x))))
    "{" * join((_json_escape(k) * ":" * json_encode(x[k]) for k in keys_sorted), ",") * "}"
end
