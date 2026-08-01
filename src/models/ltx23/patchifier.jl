# Copyright (c) 2026 Reels contributors
# SPDX-License-Identifier: Apache-2.0

const LTX23_VIDEO_SCALE_FACTORS = (8, 32, 32)

"""
Patch LTX video-VAE latents from `(channels,frames,height,width,batch)` to
`(features,tokens,batch)`.

Feature order is `(channel,local_time,local_height,local_width)` and token
order is `(frame,height,width)`, with width varying fastest.
"""
function ltx23_patchify_latents(latents::AbstractArray{T,5};
                                patch_size::NTuple{3,Int}=(1, 1, 1)) where T
    all(>(0), patch_size) ||
        throw(ArgumentError("LTX latent patch sizes must be positive"))
    channels, frames, height, width, batch = size(latents)
    pt, ph, pw = patch_size
    frames % pt == 0 && height % ph == 0 && width % pw == 0 ||
        throw(DimensionMismatch("LTX latent dimensions must divide patch size"))
    grid = (frames ÷ pt, height ÷ ph, width ÷ pw)

    split = reshape(latents, channels, pt, grid[1], ph, grid[2],
                    pw, grid[3], batch)
    ordered = permutedims(split, (6, 4, 2, 1, 7, 5, 3, 8))
    reshape(ordered, channels * pt * ph * pw, prod(grid), batch)
end

"""Undo `ltx23_patchify_latents` into `(C,F,H,W,B)`."""
function ltx23_unpatchify_latents(tokens::AbstractArray{T,3},
                                  output_shape::NTuple{5,Int};
                                  patch_size::NTuple{3,Int}=(1, 1, 1)) where T
    channels, frames, height, width, batch = output_shape
    pt, ph, pw = patch_size
    all(>(0), output_shape) ||
        throw(ArgumentError("LTX output dimensions must be positive"))
    all(>(0), patch_size) ||
        throw(ArgumentError("LTX latent patch sizes must be positive"))
    frames % pt == 0 && height % ph == 0 && width % pw == 0 ||
        throw(DimensionMismatch("LTX output dimensions must divide patch size"))
    grid = (frames ÷ pt, height ÷ ph, width ÷ pw)
    size(tokens) == (channels * pt * ph * pw, prod(grid), batch) ||
        throw(DimensionMismatch("LTX token shape does not match output shape"))

    ordered = reshape(tokens, pw, ph, pt, channels,
                      grid[3], grid[2], grid[1], batch)
    split = permutedims(ordered, (4, 3, 7, 2, 6, 1, 5, 8))
    reshape(split, output_shape)
end

"""
Return latent-grid patch bounds as `(3,tokens,2,batch)`, with axes
`(time,height,width)` and bounds `[start,end)`.
"""
function ltx23_patch_bounds(frames::Integer, height::Integer, width::Integer;
                            batch::Integer=1,
                            patch_size::NTuple{3,Int}=(1, 1, 1))
    dimensions = (Int(frames), Int(height), Int(width), Int(batch))
    all(>(0), dimensions) ||
        throw(ArgumentError("LTX latent grid dimensions must be positive"))
    all(>(0), patch_size) ||
        throw(ArgumentError("LTX latent patch sizes must be positive"))
    frames % patch_size[1] == 0 &&
        height % patch_size[2] == 0 &&
        width % patch_size[3] == 0 ||
        throw(DimensionMismatch("LTX latent grid must divide patch size"))

    tokens = (frames ÷ patch_size[1]) *
             (height ÷ patch_size[2]) *
             (width ÷ patch_size[3])
    bounds = Array{Int32}(undef, 3, tokens, 2, batch)
    token = 1
    for frame in 0:patch_size[1]:frames-1,
        row in 0:patch_size[2]:height-1,
        column in 0:patch_size[3]:width-1
        starts = (frame, row, column)
        for axis in 1:3
            bounds[axis, token, 1, :] .= starts[axis]
            bounds[axis, token, 2, :] .= starts[axis] + patch_size[axis]
        end
        token += 1
    end
    bounds
end

"""Map latent bounds to pixel-grid bounds, including LTX's causal time fix."""
function ltx23_pixel_bounds(bounds::AbstractArray{<:Real,4};
                            scale_factors::NTuple{3,Int}=
                                LTX23_VIDEO_SCALE_FACTORS,
                            causal_fix::Bool=true)
    size(bounds, 1) == 3 && size(bounds, 3) == 2 ||
        throw(DimensionMismatch("LTX bounds must have shape (3,tokens,2,batch)"))
    all(>(0), scale_factors) ||
        throw(ArgumentError("LTX scale factors must be positive"))
    result = Float32.(bounds)
    for axis in 1:3
        result[axis, :, :, :] .*= Float32(scale_factors[axis])
    end
    if causal_fix
        temporal = @view result[1, :, :, :]
        temporal .= max.(temporal .+ Float32(1 - scale_factors[1]), 0f0)
    end
    result
end

"""
Generate official LTX video RoPE midpoint coordinates as `(3,tokens,batch)`.
Temporal positions are measured in seconds.
"""
function ltx23_patch_positions(frames::Integer, height::Integer,
                               width::Integer; batch::Integer=1,
                               fps::Real=24,
                               patch_size::NTuple{3,Int}=(1, 1, 1),
                               scale_factors::NTuple{3,Int}=
                                   LTX23_VIDEO_SCALE_FACTORS,
                               causal_fix::Bool=true)
    fps > 0 || throw(ArgumentError("LTX video FPS must be positive"))
    latent_bounds = ltx23_patch_bounds(frames, height, width;
        batch=batch, patch_size=patch_size)
    pixel_bounds = ltx23_pixel_bounds(latent_bounds;
        scale_factors=scale_factors, causal_fix=causal_fix)
    positions = dropdims(sum(pixel_bounds; dims=3); dims=3) ./ 2f0
    positions[1, :, :] ./= Float32(fps)
    positions
end
