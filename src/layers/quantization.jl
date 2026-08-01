"""
Per-output-channel symmetric Int8 storage for a frozen linear weight.
`T` is the compute dtype produced when the weight is materialized.
"""
struct QuantizedMatrix{T,Q<:AbstractMatrix{Int8},S<:AbstractVector{Float32}} <:
       AbstractMatrix{T}
    values::Q
    scales::S
end

"""CPU-resident storage for a frozen linear weight."""
struct CPUOffloadedMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    values::M
end

Base.size(weight::QuantizedMatrix) = size(weight.values)
Base.size(weight::CPUOffloadedMatrix) = size(weight.values)
Base.getindex(weight::QuantizedMatrix{T}, i::Int, j::Int) where T =
    T(weight.values[i, j]) * T(weight.scales[i])
Base.getindex(weight::CPUOffloadedMatrix, i::Int, j::Int) =
    weight.values[i, j]
Base.IndexStyle(::Type{<:QuantizedMatrix}) = IndexCartesian()
Base.IndexStyle(::Type{<:CPUOffloadedMatrix}) = IndexCartesian()

function _host_dequantize(weight::QuantizedMatrix{T}) where T
    values = Float32.(Array(weight.values))
    restored = values .* reshape(Array(weight.scales), :, 1)
    cast_values(T, restored)
end
Base.Array(weight::QuantizedMatrix) = Array(_host_dequantize(weight))
Base.Array(weight::CPUOffloadedMatrix) = Array(weight.values)
Base.copy(weight::QuantizedMatrix) = _host_dequantize(weight)
Base.copy(weight::CPUOffloadedMatrix) = copy(weight.values)

function quantize_frozen_matrix(weight::AbstractMatrix,
                                ::Type{T}=eltype(weight)) where T
    host = Float32.(Array(weight))
    maxima = vec(maximum(abs, host; dims=2))
    scales = maxima ./ 127f0
    scales[maxima .== 0f0] .= 1f0
    values = Int8.(round.(clamp.(
        host ./ reshape(scales, :, 1), -127f0, 127f0)))
    QuantizedMatrix{T,typeof(values),typeof(scales)}(values, scales)
end

function _array_on_input_device(array::AbstractArray,
                                input::AbstractArray)
    if input isa CUDA.CuArray
        return array isa CUDA.CuArray ? array : CUDA.CuArray(Array(array))
    end
    array isa CUDA.CuArray ? Array(array) : array
end

function _materialize_weight(weight::QuantizedMatrix{T},
                             input::AbstractArray) where T
    values = _array_on_input_device(weight.values, input)
    scales = _array_on_input_device(weight.scales, input)
    _dequantize_for_compute(values, scales, eltype(input))
end

_dequantize_for_compute(values, scales, ::Type{Float16}) =
    Float16.(values) .* reshape(Float16.(scales), :, 1)
_dequantize_for_compute(values, scales, ::Type{Float32}) =
    Float32.(values) .* reshape(Float32.(scales), :, 1)

function _materialize_weight(weight::CPUOffloadedMatrix,
                             input::AbstractArray)
    stored = _array_on_input_device(weight.values, input)
    cast_values(eltype(input), stored)
end

_materialize_weight(weight::AbstractMatrix, ::AbstractArray) = weight
_weight_mul(weight::AbstractMatrix, input::AbstractMatrix) =
    _materialize_weight(weight, input) * input
_weight_transpose_mul(weight::AbstractMatrix, input::AbstractMatrix) =
    _materialize_weight(weight, input)' * input

struct FrozenWeightTransfer{F}
    regular::F
    device::Symbol
    precision::Symbol
    quantization::Symbol
    cpu_offload::Bool
end
(transfer::FrozenWeightTransfer)(array::AbstractArray) =
    transfer.regular(array)

function frozen_weight_transfer(device::Symbol, precision::Symbol;
                                quantization::Symbol=:none,
                                cpu_offload::Bool=false)
    quantization in (:none, :int8) ||
        throw(ArgumentError(
            "frozen weight quantization must be none or int8"))
    cpu_offload && device !== :cuda &&
        throw(ArgumentError("CPU weight offloading requires device=:cuda"))
    (quantization !== :none || cpu_offload) && precision === :bf16 &&
        throw(ArgumentError(
            "low-VRAM frozen weights currently require fp16 or fp32 compute"))
    FrozenWeightTransfer(array_transfer(device, precision), device, precision,
                         quantization, cpu_offload)
end

_move_weight(weight::AbstractMatrix, transfer) = transfer(weight)
_cpu_offloaded_matrix(weight, ::Type{Float16}) =
    CPUOffloadedMatrix{Float16,Matrix{Float16}}(
        Float16.(Array(weight)))
_cpu_offloaded_matrix(weight, ::Type{Float32}) =
    CPUOffloadedMatrix{Float32,Matrix{Float32}}(
        Float32.(Array(weight)))

function _move_weight(weight::AbstractMatrix,
                      transfer::FrozenWeightTransfer)
    T = precision_eltype(transfer.precision)
    if transfer.quantization === :int8
        quantized = quantize_frozen_matrix(weight, T)
        if transfer.cpu_offload
            return quantized
        end
        values = transfer.device === :cuda ?
            CUDA.CuArray(quantized.values) : quantized.values
        scales = transfer.device === :cuda ?
            CUDA.CuArray(quantized.scales) : quantized.scales
        return QuantizedMatrix{T,typeof(values),typeof(scales)}(
            values, scales)
    end
    transfer.cpu_offload &&
        return _cpu_offloaded_matrix(weight, T)
    transfer.regular(weight)
end

function _move_weight(weight::QuantizedMatrix,
                      transfer::FrozenWeightTransfer)
    transfer.quantization === :int8 ||
        return _move_weight(Array(weight), transfer)
    T = precision_eltype(transfer.precision)
    keep_on_cpu = transfer.cpu_offload || transfer.device === :cpu
    values = keep_on_cpu ? Array(weight.values) :
        (weight.values isa CUDA.CuArray ?
            weight.values : CUDA.CuArray(weight.values))
    scales = keep_on_cpu ? Array(weight.scales) :
        (weight.scales isa CUDA.CuArray ?
            weight.scales : CUDA.CuArray(weight.scales))
    QuantizedMatrix{T,typeof(values),typeof(scales)}(values, scales)
end
