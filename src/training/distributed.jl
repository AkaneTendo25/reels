abstract type AbstractDistributedRuntime end

struct SingleProcessRuntime <: AbstractDistributedRuntime end

struct NCCLUniqueId
    bytes::NTuple{128,UInt8}
end

mutable struct NCCLRuntime <: AbstractDistributedRuntime
    rank::Int
    world_size::Int
    local_rank::Int
    communicator::Ptr{Cvoid}
    library::Ptr{Cvoid}
    closed::Bool
end

distributed_rank(::SingleProcessRuntime) = 0
distributed_world_size(::SingleProcessRuntime) = 1
distributed_rank(runtime::NCCLRuntime) = runtime.rank
distributed_world_size(runtime::NCCLRuntime) = runtime.world_size
is_main_process(runtime::AbstractDistributedRuntime) =
    distributed_rank(runtime) == 0

function _distributed_env_int(name::AbstractString, fallback::Integer)
    value = get(ENV, name, "")
    isempty(value) ? Int(fallback) : parse(Int, value)
end

function _nccl_symbol(library, name::Symbol)
    pointer = Libdl.dlsym_e(library, name)
    pointer == C_NULL &&
        throw(ErrorException("NCCL library does not export $name"))
    pointer
end

function _nccl_error(library, code::Integer)
    pointer = ccall(_nccl_symbol(library, :ncclGetErrorString),
                    Cstring, (Cint,), Cint(code))
    pointer == C_NULL ? "unknown NCCL error" : unsafe_string(pointer)
end

function _nccl_check(library, code::Integer, operation::AbstractString)
    code == 0 ||
        throw(ErrorException("$operation failed: $(_nccl_error(library, code))"))
    nothing
end

function _nccl_unique_id(library)
    reference = Ref(NCCLUniqueId(ntuple(_ -> UInt8(0), 128)))
    code = ccall(_nccl_symbol(library, :ncclGetUniqueId), Cint,
                 (Ref{NCCLUniqueId},), reference)
    _nccl_check(library, code, "ncclGetUniqueId")
    reference[]
end

_unique_id_hex(id::NCCLUniqueId) =
    join(string(byte; base=16, pad=2) for byte in id.bytes)

function _parse_unique_id(encoded::AbstractString)
    value = strip(encoded)
    ncodeunits(value) == 256 ||
        throw(ArgumentError(
            "NCCL unique id must contain 256 hexadecimal characters"))
    bytes = ntuple(128) do index
        parse(UInt8, value[2index - 1:2index]; base=16)
    end
    NCCLUniqueId(bytes)
end

function _connect_rendezvous(host::AbstractString, port::Integer,
                             timeout_seconds::Real)
    deadline = time() + Float64(timeout_seconds)
    last_error = nothing
    while time() < deadline
        try
            return Sockets.connect(String(host), Int(port))
        catch error
            last_error = error
            sleep(0.1)
        end
    end
    suffix = last_error === nothing ? "" :
        ": $(sprint(showerror, last_error))"
    throw(ErrorException(
        "timed out connecting to distributed rendezvous $host:$port$suffix"))
end

function _rendezvous_unique_id(library, rank::Integer, world_size::Integer,
                               host::AbstractString, port::Integer,
                               timeout_seconds::Real)
    supplied = get(ENV, "REELS_NCCL_UNIQUE_ID", "")
    isempty(supplied) || return _parse_unique_id(supplied)
    if rank == 0
        id = _nccl_unique_id(library)
        server = Sockets.listen(ip"0.0.0.0", Int(port))
        try
            observed = Set{Int}()
            for _ in 1:world_size-1
                socket = accept(server)
                try
                    peer_rank = parse(Int, strip(readline(socket)))
                    0 < peer_rank < world_size ||
                        throw(ArgumentError(
                            "invalid distributed peer rank $peer_rank"))
                    peer_rank in observed &&
                        throw(ArgumentError(
                            "duplicate distributed peer rank $peer_rank"))
                    push!(observed, peer_rank)
                    println(socket, _unique_id_hex(id))
                    flush(socket)
                finally
                    close(socket)
                end
            end
        finally
            close(server)
        end
        return id
    end
    socket = _connect_rendezvous(host, port, timeout_seconds)
    try
        println(socket, rank)
        flush(socket)
        _parse_unique_id(readline(socket))
    finally
        close(socket)
    end
end

function _open_nccl_library()
    requested = get(ENV, "REELS_NCCL_LIBRARY", NCCL_jll.libnccl)
    library = Libdl.dlopen_e(requested)
    library == C_NULL &&
        throw(ErrorException(
            "unable to load NCCL library '$requested'; set REELS_NCCL_LIBRARY"))
    library
end

"""
    init_distributed(config=DistributedConfig())

Initialize one NCCL process per GPU. Dynamic rank settings default to the
standard `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
environment variables. Rank zero distributes the NCCL unique id over a TCP
rendezvous; launchers may instead provide `REELS_NCCL_UNIQUE_ID`.
"""
function init_distributed(config::DistributedConfig=DistributedConfig())
    config.enabled || return SingleProcessRuntime()
    config.backend === :nccl ||
        throw(ArgumentError("only the NCCL distributed backend is supported"))
    world_size = config.world_size > 0 ? config.world_size :
        _distributed_env_int("WORLD_SIZE", 1)
    rank = config.rank >= 0 ? config.rank :
        _distributed_env_int("RANK", 0)
    local_rank = config.local_rank >= 0 ? config.local_rank :
        _distributed_env_int("LOCAL_RANK", rank)
    world_size > 1 ||
        throw(ArgumentError("distributed world size must be greater than one"))
    0 <= rank < world_size ||
        throw(ArgumentError(
            "distributed rank $rank is outside world size $world_size"))
    local_rank >= 0 ||
        throw(ArgumentError("distributed local rank must be nonnegative"))
    CUDA.functional() ||
        throw(ErrorException("NCCL distributed training requires functional CUDA"))
    CUDA.device!(local_rank)
    library = _open_nccl_library()
    master_addr = get(ENV, "MASTER_ADDR", config.master_addr)
    master_port = _distributed_env_int("MASTER_PORT", config.master_port)
    id = try
        _rendezvous_unique_id(library, rank, world_size, master_addr,
                              master_port, config.timeout_seconds)
    catch
        Libdl.dlclose(library)
        rethrow()
    end
    communicator = Ref{Ptr{Cvoid}}(C_NULL)
    code = ccall(_nccl_symbol(library, :ncclCommInitRank), Cint,
                 (Ref{Ptr{Cvoid}}, Cint, NCCLUniqueId, Cint),
                 communicator, Cint(world_size), id, Cint(rank))
    try
        _nccl_check(library, code, "ncclCommInitRank")
    catch
        Libdl.dlclose(library)
        rethrow()
    end
    NCCLRuntime(rank, world_size, local_rank, communicator[], library, false)
end

_nccl_datatype(::Type{Float16}) = Cint(6)
_nccl_datatype(::Type{Float32}) = Cint(7)
_nccl_datatype(::Type{Float64}) = Cint(8)
_nccl_datatype(::Type{BFloat16}) = Cint(9)
_nccl_datatype(::Type{T}) where T =
    throw(ArgumentError("NCCL all-reduce does not support $T"))

function _nccl_allreduce_sum!(runtime::NCCLRuntime, array::CUDA.CuArray)
    runtime.closed &&
        throw(ArgumentError("NCCL runtime is closed"))
    device_pointer = convert(CUDA.CuPtr{Cvoid}, pointer(array))
    stream_pointer = Ptr{Cvoid}(getfield(CUDA.stream(), :handle))
    code = ccall(_nccl_symbol(runtime.library, :ncclAllReduce), Cint,
                 (CUDA.CuPtr{Cvoid}, CUDA.CuPtr{Cvoid}, Csize_t,
                  Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}),
                 device_pointer, device_pointer, Csize_t(length(array)),
                 _nccl_datatype(eltype(array)), Cint(0),
                 runtime.communicator, stream_pointer)
    _nccl_check(runtime.library, code, "ncclAllReduce")
    array
end

allreduce_mean!(::SingleProcessRuntime, array::AbstractArray) = array
function allreduce_mean!(runtime::NCCLRuntime, array::CUDA.CuArray)
    _nccl_allreduce_sum!(runtime, array)
    CUDA.synchronize()
    array .*= Float32(inv(runtime.world_size))
    array
end
allreduce_mean!(::NCCLRuntime, ::AbstractArray) =
    throw(ArgumentError("NCCL all-reduce requires CUDA arrays"))

allreduce_gradients!(::SingleProcessRuntime, gradients) = gradients
function allreduce_gradients!(runtime::NCCLRuntime, gradients)
    all(gradient -> gradient isa CUDA.CuArray, gradients) ||
        throw(ArgumentError("NCCL gradients must all be CUDA arrays"))
    start_code = ccall(_nccl_symbol(runtime.library, :ncclGroupStart),
                       Cint, ())
    _nccl_check(runtime.library, start_code, "ncclGroupStart")
    try
        foreach(gradient -> _nccl_allreduce_sum!(runtime, gradient), gradients)
    finally
        end_code = ccall(_nccl_symbol(runtime.library, :ncclGroupEnd),
                         Cint, ())
        _nccl_check(runtime.library, end_code, "ncclGroupEnd")
    end
    CUDA.synchronize()
    scale = Float32(inv(runtime.world_size))
    foreach(gradient -> (gradient .*= scale), gradients)
    gradients
end

distributed_mean_scalar(::SingleProcessRuntime, value::Real, like=nothing) =
    Float32(value)
function distributed_mean_scalar(runtime::NCCLRuntime, value::Real,
                                 like::CUDA.CuArray)
    buffer = similar(like, Float32, 1)
    fill!(buffer, Float32(value))
    allreduce_mean!(runtime, buffer)
    only(Array(buffer))
end

distributed_barrier!(::SingleProcessRuntime) = nothing
function distributed_barrier!(runtime::NCCLRuntime)
    buffer = CUDA.zeros(Float32, 1)
    _nccl_allreduce_sum!(runtime, buffer)
    CUDA.synchronize()
    nothing
end

close_distributed!(::SingleProcessRuntime) = nothing
function close_distributed!(runtime::NCCLRuntime)
    runtime.closed && return nothing
    code = ccall(_nccl_symbol(runtime.library, :ncclCommDestroy),
                 Cint, (Ptr{Cvoid},), runtime.communicator)
    _nccl_check(runtime.library, code, "ncclCommDestroy")
    runtime.closed = true
    runtime.communicator = C_NULL
    Libdl.dlclose(runtime.library)
    runtime.library = C_NULL
    nothing
end
