struct CachedBatchProvider
    entries::Vector{String}
    batch_size::Int
    device::Symbol
    precision::Symbol
    rank::Int
    world_size::Int
end

struct LTXCachedBatchProvider
    entries::Vector{String}
    batch_size::Int
    device::Symbol
    precision::Symbol
    rank::Int
    world_size::Int
end

"""
    distributed_sample_indices(rng, count, batch_size, rank, world_size)

Draw one global permutation and return this rank's disjoint contiguous shard.
All ranks must start from the same RNG state, which is preserved by Reels
checkpoint/resume. This makes each optimizer micro-step consume
`batch_size * world_size` distinct cache entries.
"""
function distributed_sample_indices(rng::AbstractRNG, count::Integer,
                                    batch_size::Integer, rank::Integer,
                                    world_size::Integer)
    count > 0 || throw(ArgumentError("sample count must be positive"))
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    world_size > 0 || throw(ArgumentError("world size must be positive"))
    0 <= rank < world_size ||
        throw(ArgumentError("rank must be in 0:world_size-1"))
    required = batch_size * world_size
    required <= count ||
        throw(ArgumentError(
            "distributed batch requires $required entries, found $count"))
    permutation = randperm(rng, Int(count))
    offset = Int(rank) * Int(batch_size)
    permutation[offset + 1:offset + Int(batch_size)]
end

function LTXCachedBatchProvider(entries::AbstractVector{<:AbstractString};
                                batch_size::Integer=1, device=:cpu,
                                precision=:fp32, rank::Integer=0,
                                world_size::Integer=1)
    isempty(entries) && throw(ArgumentError("LTX cache entry list is empty"))
    0 < batch_size * world_size <= length(entries) ||
        throw(ArgumentError("invalid LTX cache batch size"))
    device in (:cpu, :cuda) ||
        throw(ArgumentError("LTX cache device must be :cpu or :cuda"))
    precision_eltype(precision)
    0 <= rank < world_size ||
        throw(ArgumentError("invalid LTX distributed rank"))
    LTXCachedBatchProvider(String.(entries), Int(batch_size),
                           device, precision, Int(rank), Int(world_size))
end

function (provider::LTXCachedBatchProvider)(rng::AbstractRNG)
    selected = distributed_sample_indices(
        rng, length(provider.entries), provider.batch_size,
        provider.rank, provider.world_size)
    loaded = [load_ltx23_preprocess_cache(provider.entries[index])
              for index in selected]
    latent_shape = size(first(loaded).latents)
    context_shape = size(first(loaded).text_context)
    position_shape = size(first(loaded).positions)
    all(item -> size(item.latents) == latent_shape, loaded) ||
        throw(DimensionMismatch("selected LTX latents use different buckets"))
    all(item -> size(item.text_context) == context_shape, loaded) ||
        throw(DimensionMismatch("selected LTX contexts have different shapes"))
    all(item -> size(item.positions) == position_shape, loaded) ||
        throw(DimensionMismatch("selected LTX positions use different buckets"))
    transfer = array_transfer(provider.device, provider.precision)
    latents = transfer(cat((item.latents for item in loaded)...; dims=3))
    context = transfer(cat((item.text_context for item in loaded)...; dims=3))
    positions = cat((item.positions for item in loaded)...; dims=3)
    LTXLatentBatch(latents, context, positions)
end

function CachedBatchProvider(entries::AbstractVector{<:AbstractString};
                             batch_size::Integer=1, device=:cpu,
                             precision=:fp32, rank::Integer=0,
                             world_size::Integer=1)
    isempty(entries) && throw(ArgumentError("cache entry list is empty"))
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    batch_size * world_size <= length(entries) ||
        throw(ArgumentError("batch size exceeds cache entry count"))
    device in (:cpu, :cuda) ||
        throw(ArgumentError("cache provider device must be :cpu or :cuda"))
    precision_eltype(precision)
    0 <= rank < world_size ||
        throw(ArgumentError("invalid distributed cache rank"))
    CachedBatchProvider(String.(entries), Int(batch_size), device, precision,
                        Int(rank), Int(world_size))
end

function (provider::CachedBatchProvider)(rng::AbstractRNG)
    selected = distributed_sample_indices(
        rng, length(provider.entries), provider.batch_size,
        provider.rank, provider.world_size)
    loaded = [load_preprocess_cache(provider.entries[index])
              for index in selected]
    latent_shape = size(first(loaded).latents)
    text_shape = size(first(loaded).text_context)
    all(item -> size(item.latents) == latent_shape, loaded) ||
        throw(DimensionMismatch("selected latent cache entries use different buckets"))
    all(item -> size(item.text_context) == text_shape, loaded) ||
        throw(DimensionMismatch("selected text cache entries have different shapes"))
    i2v = first(loaded).conditioning_video !== nothing
    all(item -> (item.conditioning_video !== nothing) == i2v,
        loaded) ||
        throw(DimensionMismatch("cannot mix T2V and I2V cache entries"))
    if i2v
        conditioning_shape = size(first(loaded).conditioning_video)
        image_shape = size(first(loaded).image_features)
        all(item -> size(item.conditioning_video) == conditioning_shape,
            loaded) ||
            throw(DimensionMismatch(
                "selected I2V conditioning entries use different buckets"))
        all(item -> size(item.image_features) == image_shape, loaded) ||
            throw(DimensionMismatch(
                "selected I2V image features have different shapes"))
    end
    latents = cat((item.latents for item in loaded)...; dims=5)
    text_context = cat((item.text_context for item in loaded)...; dims=3)
    transfer = array_transfer(provider.device, provider.precision)
    latents, text_context = transfer(latents), transfer(text_context)
    conditioning = i2v ? transfer(cat(
        (item.conditioning_video for item in loaded)...; dims=5)) : nothing
    image_features = i2v ? transfer(cat(
        (item.image_features for item in loaded)...; dims=3)) : nothing
    WanLatentBatch(latents, text_context;
        conditioning_video=conditioning,
        image_features=image_features)
end
