"""Interface implemented by every video diffusion model family."""
abstract type AbstractVideoModelBackend end

function precision_eltype(precision::Symbol)
    precision === :fp32 && return Float32
    precision === :fp16 && return Float16
    precision === :bf16 && return BFloat16
    throw(ArgumentError("unsupported compute precision: $precision"))
end

# BFloat16s conversions currently lower incorrectly in CUDA broadcast kernels
# on some Julia/PTX combinations. Convert through the IEEE bit layout instead.
function float32_values(array::AbstractArray{BFloat16})
    words = UInt32.(reinterpret(UInt16, array)) .<< 16
    reinterpret(Float32, words)
end
float32_values(array::AbstractArray) = Float32.(array)

function bfloat16_values(array::AbstractArray)
    values = Float32.(array)
    bits = reinterpret(UInt32, values)
    rounded = bits .+ UInt32(0x00007fff) .+
              ((bits .>> 16) .& UInt32(1))
    reinterpret(BFloat16, UInt16.(rounded .>> 16))
end

cast_values(::Type{BFloat16}, array::AbstractArray) =
    eltype(array) === BFloat16 ? array : bfloat16_values(array)
cast_values(::Type{Float32}, array::AbstractArray{BFloat16}) =
    float32_values(array)
cast_values(::Type{T}, array::AbstractArray) where {T} =
    eltype(array) === T ? array : T.(array)

function array_transfer(device::Symbol, precision::Symbol)
    device in (:cpu, :cuda) ||
        throw(ArgumentError("device must be :cpu or :cuda"))
    T = precision_eltype(precision)
    if device === :cpu
        precision === :bf16 &&
            throw(ArgumentError("BF16 compute is supported on CUDA devices; " *
                                "use fp32 for CPU execution"))
        return array -> T.(Array(array))
    end
    CUDA.functional() ||
        throw(ArgumentError("CUDA is not functional on this host"))
    return function (array)
        device_array = CUDA.CuArray(Array(array))
        cast_values(T, device_array)
    end
end

for fn in (:model_family, :load_config, :load_transformer, :load_text_encoder,
           :load_vae, :load_image_encoder, :encode_text, :encode_image,
           :encode_video,
           :prepare_conditioning,
           :decode_video,
           :sample_training_target, :predict_velocity, :lora_targets,
           :export_mapping)
    @eval function $fn(backend::AbstractVideoModelBackend, args...; kwargs...)
        throw(MethodError($fn, (backend, args...)))
    end
end
