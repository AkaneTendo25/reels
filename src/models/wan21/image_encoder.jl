struct WanCLIPVisionConfig
    image_size::Int
    patch_size::Int
    hidden_size::Int
    intermediate_size::Int
    heads::Int
    layers::Int
    output_layer::Int
    epsilon::Float32
end

function WanCLIPVisionConfig(; image_size=224, patch_size=14,
                             hidden_size=1280, intermediate_size=5120,
                             heads=16, layers=32,
                             output_layer=layers - 1, epsilon=1f-5)
    image_size % patch_size == 0 ||
        throw(ArgumentError("CLIP image_size must be divisible by patch_size"))
    hidden_size % heads == 0 ||
        throw(ArgumentError("CLIP hidden_size must be divisible by heads"))
    1 <= output_layer <= layers ||
        throw(ArgumentError("CLIP output_layer must be in 1:layers"))
    WanCLIPVisionConfig(image_size, patch_size, hidden_size,
        intermediate_size, heads, layers, output_layer, Float32(epsilon))
end

struct WanCLIPLayerNorm{W<:AbstractVector,B<:AbstractVector}
    weight::W
    bias::B
    epsilon::Float32
end

(norm::WanCLIPLayerNorm)(x::AbstractArray) =
    layernorm(x; epsilon=norm.epsilon, weight=norm.weight, bias=norm.bias)

struct WanCLIPAttention
    q::DenseLayer
    k::DenseLayer
    v::DenseLayer
    o::DenseLayer
    heads::Int
end

struct WanCLIPMLP
    fc1::DenseLayer
    fc2::DenseLayer
end

struct WanCLIPEncoderLayer
    norm1::WanCLIPLayerNorm
    attention::WanCLIPAttention
    norm2::WanCLIPLayerNorm
    mlp::WanCLIPMLP
end

struct WanCLIPVisionEncoder
    config::WanCLIPVisionConfig
    class_embedding::AbstractVector
    position_embedding::AbstractMatrix
    patch_weight::AbstractArray
    pre_norm::WanCLIPLayerNorm
    layers::Vector{WanCLIPEncoderLayer}
    post_norm::WanCLIPLayerNorm
end

function WanCLIPVisionEncoder(config::WanCLIPVisionConfig;
                              rng=Random.default_rng())
    d, f = config.hidden_size, config.intermediate_size
    dense(output, input) = DenseLayer(
        randn(rng, Float32, output, input) .* 0.02f0,
        zeros(Float32, output))
    norm() = WanCLIPLayerNorm(
        ones(Float32, d), zeros(Float32, d), config.epsilon)
    layers = [WanCLIPEncoderLayer(norm(),
        WanCLIPAttention(dense(d, d), dense(d, d), dense(d, d),
                         dense(d, d), config.heads),
        norm(), WanCLIPMLP(dense(f, d), dense(d, f)))
        for _ in 1:config.layers]
    patches = (config.image_size ÷ config.patch_size)^2
    WanCLIPVisionEncoder(config,
        randn(rng, Float32, d) .* 0.02f0,
        randn(rng, Float32, d, patches + 1) .* 0.02f0,
        randn(rng, Float32, d, 3, config.patch_size,
              config.patch_size) .* 0.02f0,
        norm(), layers, norm())
end

function wan_clip_vision_specs(config::WanCLIPVisionConfig=
                               WanCLIPVisionConfig())
    d, f, p = config.hidden_size, config.intermediate_size, config.patch_size
    tokens = (config.image_size ÷ p)^2 + 1
    specs = TensorSpec[
        TensorSpec("vision_model.embeddings.class_embedding", [d],
            VECTOR_LAYOUT),
        TensorSpec("vision_model.embeddings.patch_embedding.weight",
            [d, 3, p, p], CONV_OUT_IN_SPATIAL),
        TensorSpec("vision_model.embeddings.position_embedding.weight",
            [tokens, d], ROW_MAJOR_SOURCE;
            destination="vision_model.embeddings.position_embedding.weight"),
        TensorSpec("vision_model.pre_layrnorm.weight", [d], VECTOR_LAYOUT),
        TensorSpec("vision_model.pre_layrnorm.bias", [d], VECTOR_LAYOUT),
    ]
    for index in 0:config.layers - 1
        prefix = "vision_model.encoder.layers.$index"
        for norm_name in ("layer_norm1", "layer_norm2")
            push!(specs,
                TensorSpec("$prefix.$norm_name.weight", [d], VECTOR_LAYOUT),
                TensorSpec("$prefix.$norm_name.bias", [d], VECTOR_LAYOUT))
        end
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
            push!(specs,
                TensorSpec("$prefix.self_attn.$projection.weight",
                    [d, d], LINEAR_OUT_IN),
                TensorSpec("$prefix.self_attn.$projection.bias",
                    [d], VECTOR_LAYOUT))
        end
        push!(specs,
            TensorSpec("$prefix.mlp.fc1.weight", [f, d], LINEAR_OUT_IN),
            TensorSpec("$prefix.mlp.fc1.bias", [f], VECTOR_LAYOUT),
            TensorSpec("$prefix.mlp.fc2.weight", [d, f], LINEAR_OUT_IN),
            TensorSpec("$prefix.mlp.fc2.bias", [d], VECTOR_LAYOUT))
    end
    push!(specs,
        TensorSpec("vision_model.post_layernorm.weight", [d], VECTOR_LAYOUT),
        TensorSpec("vision_model.post_layernorm.bias", [d], VECTOR_LAYOUT),
        # CLIPVisionModelWithProjection checkpoints include this head. Wan
        # consumes hidden patch tokens, so the projection is recognized but
        # intentionally not loaded into the runtime encoder.
        TensorSpec("visual_projection.weight", [1024, d], LINEAR_OUT_IN;
            required=false))
    specs
end

function _wan_clip_softmax(scores)
    values = float32_values(scores)
    exponentials = exp.(values .- maximum(values; dims=1))
    cast_values(eltype(scores), exponentials ./ sum(exponentials; dims=1))
end

function wan_clip_attention_forward(attention::WanCLIPAttention,
                                    x::AbstractArray)
    d, tokens, batch = size(x)
    head_size = d ÷ attention.heads
    q = reshape(linear(attention.q, x),
        head_size, attention.heads, tokens, batch)
    k = reshape(linear(attention.k, x),
        head_size, attention.heads, tokens, batch)
    v = reshape(linear(attention.v, x),
        head_size, attention.heads, tokens, batch)
    scores = similar(q, eltype(q), tokens, tokens, attention.heads, batch)
    scale = inv(sqrt(Float32(head_size)))
    for b in 1:batch, h in 1:attention.heads
        scores[:, :, h, b] .=
            (transpose(view(k, :, h, :, b)) *
             view(q, :, h, :, b)) .* scale
    end
    probabilities = _wan_clip_softmax(scores)
    attended = similar(q)
    for b in 1:batch, h in 1:attention.heads
        attended[:, h, :, b] .=
            view(v, :, h, :, b) * view(probabilities, :, :, h, b)
    end
    linear(attention.o, reshape(attended, d, tokens, batch))
end

function wan_clip_layer_forward(layer::WanCLIPEncoderLayer,
                                x::AbstractArray)
    x = mixed_add(x, wan_clip_attention_forward(layer.attention,
                                                 layer.norm1(x)))
    hidden = linear(layer.mlp.fc1, layer.norm2(x))
    # Hugging Face CLIP's `hidden_act = "gelu"` is the exact erf form, not
    # NNlib's default tanh approximation.
    hidden32 = float32_values(hidden)
    activated = 0.5f0 .* hidden32 .*
        (1f0 .+ SpecialFunctions.erf.(hidden32 ./ sqrt(2f0)))
    hidden = cast_values(eltype(hidden), activated)
    mixed_add(x, linear(layer.mlp.fc2, hidden))
end

function _wan_clip_patchify(model::WanCLIPVisionEncoder,
                            images::AbstractArray)
    ndims(images) == 4 ||
        throw(DimensionMismatch("CLIP images must be (C,H,W,B)"))
    size(images, 1) == 3 ||
        throw(DimensionMismatch("CLIP images must have three channels"))
    size(images, 2) == model.config.image_size &&
        size(images, 3) == model.config.image_size ||
        throw(DimensionMismatch(
            "CLIP images must be $(model.config.image_size)x" *
            "$(model.config.image_size)"))
    # NNlib convolution uses (W,H,C,N) and (W,H,C,O).
    input = permutedims(images, (3, 2, 1, 4))
    # NNlib performs mathematical convolution; PyTorch CLIP's Conv2d uses
    # cross-correlation, so reverse both spatial kernel axes.
    kernel = reverse(permutedims(model.patch_weight, (4, 3, 2, 1));
                     dims=(1, 2))
    patches = NNlib.conv(input, kernel;
        stride=(model.config.patch_size, model.config.patch_size))
    width, height, d, batch = size(patches)
    reshape(permutedims(patches, (3, 1, 2, 4)),
        d, width * height, batch)
end

function wan_clip_vision_forward(model::WanCLIPVisionEncoder,
                                 images::AbstractArray)
    batched = ndims(images) == 3 ?
        reshape(images, size(images)..., 1) : images
    patches = _wan_clip_patchify(model, batched)
    batch = size(patches, 3)
    class_token = repeat(reshape(model.class_embedding, :, 1, 1),
                         1, 1, batch)
    hidden = cat(class_token, patches; dims=2)
    hidden = mixed_add(hidden, reshape(
        model.position_embedding, size(hidden, 1), size(hidden, 2), 1))
    hidden = model.pre_norm(hidden)
    for index in 1:model.config.output_layer
        hidden = wan_clip_layer_forward(model.layers[index], hidden)
    end
    ndims(images) == 3 ? dropdims(hidden; dims=3) : hidden
end

function load_wan_clip_vision(source::AbstractTensorSource,
                              config::WanCLIPVisionConfig=
                                  WanCLIPVisionConfig();
                              strict=true)
    specs = wan_clip_vision_specs(config)
    audit = audit_state_dict(source, specs; allow_unexpected=!strict)
    isempty(audit) || throw(ArgumentError(
        "Wan CLIP state dictionary mismatch: missing=$(audit.missing), " *
        "unexpected=$(audit.unexpected), shapes=$(audit.shape_mismatches)"))
    get_tensor(key) = load_state_tensor(source,
        only(filter(spec -> spec.source_key == key, specs)))
    d = config.hidden_size
    norm(prefix) = WanCLIPLayerNorm(
        get_tensor("$prefix.weight"), get_tensor("$prefix.bias"),
        config.epsilon)
    dense(prefix) = DenseLayer(
        get_tensor("$prefix.weight"), get_tensor("$prefix.bias"))
    layers = WanCLIPEncoderLayer[]
    for index in 0:config.layers - 1
        prefix = "vision_model.encoder.layers.$index"
        push!(layers, WanCLIPEncoderLayer(
            norm("$prefix.layer_norm1"),
            WanCLIPAttention(
                dense("$prefix.self_attn.q_proj"),
                dense("$prefix.self_attn.k_proj"),
                dense("$prefix.self_attn.v_proj"),
                dense("$prefix.self_attn.out_proj"), config.heads),
            norm("$prefix.layer_norm2"),
            WanCLIPMLP(dense("$prefix.mlp.fc1"),
                       dense("$prefix.mlp.fc2"))))
    end
    position = permutedims(get_tensor(
        "vision_model.embeddings.position_embedding.weight"))
    WanCLIPVisionEncoder(config,
        get_tensor("vision_model.embeddings.class_embedding"),
        position,
        get_tensor("vision_model.embeddings.patch_embedding.weight"),
        norm("vision_model.pre_layrnorm"), layers,
        norm("vision_model.post_layernorm"))
end

load_wan_clip_vision(path::AbstractString,
                     config::WanCLIPVisionConfig=WanCLIPVisionConfig();
                     kwargs...) =
    load_wan_clip_vision(open_tensor_source(path), config; kwargs...)

function load_image_encoder(::Wan21, checkpoint::AbstractString;
                            device=:cpu, precision=:fp32,
                            config=WanCLIPVisionConfig(), strict=true)
    device in (:cpu, :cuda) ||
        throw(ArgumentError(
            "Wan image encoder device must be :cpu or :cuda"))
    precision in (:fp32, :fp16, :bf16) ||
        throw(ArgumentError(
            "unsupported Wan image encoder precision $precision"))
    encoder = load_wan_clip_vision(checkpoint, config; strict=strict)
    move_to_device(encoder, device, precision)
end

function wan_clip_vision_state_dict(model::WanCLIPVisionEncoder)
    state = Dict{String,AbstractArray}(
        "vision_model.embeddings.class_embedding" =>
            Array(model.class_embedding),
        "vision_model.embeddings.patch_embedding.weight" =>
            Array(model.patch_weight),
        "vision_model.embeddings.position_embedding.weight" =>
            permutedims(Array(model.position_embedding)),
        "vision_model.pre_layrnorm.weight" =>
            Array(model.pre_norm.weight),
        "vision_model.pre_layrnorm.bias" =>
            Array(model.pre_norm.bias),
        "vision_model.post_layernorm.weight" =>
            Array(model.post_norm.weight),
        "vision_model.post_layernorm.bias" =>
            Array(model.post_norm.bias),
    )
    for (offset, layer) in enumerate(model.layers)
        prefix = "vision_model.encoder.layers.$(offset - 1)"
        state["$prefix.layer_norm1.weight"] = Array(layer.norm1.weight)
        state["$prefix.layer_norm1.bias"] = Array(layer.norm1.bias)
        state["$prefix.layer_norm2.weight"] = Array(layer.norm2.weight)
        state["$prefix.layer_norm2.bias"] = Array(layer.norm2.bias)
        for (name, dense) in (
            "q_proj" => layer.attention.q,
            "k_proj" => layer.attention.k,
            "v_proj" => layer.attention.v,
            "out_proj" => layer.attention.o)
            state["$prefix.self_attn.$name.weight"] = Array(dense.weight)
            state["$prefix.self_attn.$name.bias"] = Array(dense.bias)
        end
        state["$prefix.mlp.fc1.weight"] = Array(layer.mlp.fc1.weight)
        state["$prefix.mlp.fc1.bias"] = Array(layer.mlp.fc1.bias)
        state["$prefix.mlp.fc2.weight"] = Array(layer.mlp.fc2.weight)
        state["$prefix.mlp.fc2.bias"] = Array(layer.mlp.fc2.bias)
    end
    state
end

function _move_wan_clip_norm(norm::WanCLIPLayerNorm, transfer)
    WanCLIPLayerNorm(transfer(norm.weight), transfer(norm.bias), norm.epsilon)
end

function move_to_device(model::WanCLIPVisionEncoder, device::Symbol,
                        precision::Symbol=:fp32)
    transfer = array_transfer(device, precision)
    dense(layer) = DenseLayer(
        transfer(layer.weight), transfer(layer.bias))
    layers = [WanCLIPEncoderLayer(
        _move_wan_clip_norm(layer.norm1, transfer),
        WanCLIPAttention(dense(layer.attention.q), dense(layer.attention.k),
            dense(layer.attention.v), dense(layer.attention.o),
            layer.attention.heads),
        _move_wan_clip_norm(layer.norm2, transfer),
        WanCLIPMLP(dense(layer.mlp.fc1), dense(layer.mlp.fc2)))
        for layer in model.layers]
    WanCLIPVisionEncoder(model.config, transfer(model.class_embedding),
        transfer(model.position_embedding), transfer(model.patch_weight),
        _move_wan_clip_norm(model.pre_norm, transfer), layers,
        _move_wan_clip_norm(model.post_norm, transfer))
end

const WAN_CLIP_MEAN = Float32[0.48145466, 0.4578275, 0.40821073]
const WAN_CLIP_STD = Float32[0.26862954, 0.26130258, 0.27577711]

function _wan_clip_cubic(x::Float64)
    distance = abs(x)
    a = -0.5
    distance < 1 &&
        return ((a + 2) * distance - (a + 3)) *
               distance * distance + 1
    distance < 2 &&
        return (((a * distance - 5a) * distance + 8a) *
                distance - 4a)
    0.0
end

function _wan_clip_resample_coefficients(source::Int, destination::Int)
    scale = destination / source
    filter_scale = min(scale, 1.0)
    support = 2.0 / filter_scale
    map(1:destination) do output_index
        # Matches Pillow's half-pixel center convention and widens the cubic
        # support during reduction to provide antialiasing.
        center = (output_index - 0.5) / scale
        first_source = max(0, floor(Int, center - support + 0.5))
        last_source = min(source, floor(Int, center + support + 0.5))
        indices = collect(first_source:last_source - 1)
        weights = Float64[
            _wan_clip_cubic(
                (index - center + 0.5) * filter_scale) * filter_scale
            for index in indices]
        total = sum(weights)
        total == 0 || (weights ./= total)
        (indices .+ 1, weights)
    end
end

function _bicubic_resize_chw(image::AbstractArray, height::Int, width::Int)
    channels, source_height, source_width = size(image)
    horizontal = _wan_clip_resample_coefficients(source_width, width)
    vertical = _wan_clip_resample_coefficients(source_height, height)
    temporary = zeros(Float64, channels, source_height, width)
    for x in 1:width
        indices, weights = horizontal[x]
        for (index, weight) in zip(indices, weights)
            @views temporary[:, :, x] .+= weight .* image[:, :, index]
        end
    end
    output = zeros(Float64, channels, height, width)
    for y in 1:height
        indices, weights = vertical[y]
        for (index, weight) in zip(indices, weights)
            @views output[:, y, :] .+= weight .* temporary[:, index, :]
        end
    end
    output
end

function preprocess_wan_clip_image(image::AbstractArray;
                                   image_size::Int=224)
    ndims(image) == 3 && size(image, 1) == 3 ||
        throw(DimensionMismatch("CLIP image must be (3,H,W)"))
    values = Float32.(Array(image))
    # Video decoding supplies [-1,1], while callers may also supply [0,1].
    minimum(values) < 0f0 && (values = (values .+ 1f0) ./ 2f0)
    values = clamp.(values, 0f0, 1f0)
    # The official Wan Diffusers image processor sets an explicit 224×224
    # resize and `do_center_crop=false`; non-square references are resized
    # directly instead of preserving aspect ratio.
    resized = _bicubic_resize_chw(values, image_size, image_size)
    # The official processor resizes a Pillow RGB image. Its uint8 output
    # clamps cubic overshoot and rounds to byte precision before rescaling.
    resized = Float32.(
        round.(clamp.(resized, 0.0, 1.0) .* 255.0) ./ 255.0)
    (resized .- reshape(WAN_CLIP_MEAN, 3, 1, 1)) ./
        reshape(WAN_CLIP_STD, 3, 1, 1)
end

function encode_image(::Wan21, encoder::WanCLIPVisionEncoder,
                      image::AbstractArray)
    processed = preprocess_wan_clip_image(
        image; image_size=encoder.config.image_size)
    on_cuda = encoder.patch_weight isa CUDA.CuArray
    input = on_cuda ? CUDA.CuArray(processed) : processed
    wan_clip_vision_forward(encoder, input)
end
