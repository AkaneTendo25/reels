struct SyntheticBackend <: AbstractVideoModelBackend
    input_dim::Int
    output_dim::Int
end
model_family(::SyntheticBackend) = :synthetic

mutable struct SyntheticModel
    projection::LoRALinear
end

function load_transformer(b::SyntheticBackend, checkpoint=nothing, device=:cpu,
                          precision=:fp32; rank=4, rng=Xoshiro(1))
    weight = randn(rng, Float32, b.output_dim, b.input_dim) ./ sqrt(Float32(b.input_dim))
    SyntheticModel(LoRALinear(weight; rank=rank, rng=rng))
end
predict_velocity(::SyntheticBackend, model::SyntheticModel, noisy, conditioning=nothing) =
    model.projection(noisy)
lora_targets(::SyntheticBackend, model::SyntheticModel, selection=:all) =
    lora_parameters(model.projection)
export_mapping(::SyntheticBackend) = Dict("projection.lora_A.weight" => :A,
    "projection.lora_B.weight" => :B)

function synthetic_batch(rng, input_dim, output_dim, n; teacher=nothing)
    x = randn(rng, Float32, input_dim, n)
    target = teacher === nothing ? 2f0 .* x[1:output_dim, :] : teacher * x
    (x=x, target=target)
end
