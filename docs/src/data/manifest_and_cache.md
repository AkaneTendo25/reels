# Captioned-video manifests and preprocessing cache

Reels accepts strict TOML and JSONL manifests.

```toml
[[samples]]
video = "clips/00001.mp4"
caption = "A camera tracks a cyclist through rain."

[[samples]]
video = "clips/00002.mp4"
caption_file = "clips/00002.txt"
```

Relative paths resolve against the manifest file. Every sample must provide
exactly one caption source, unknown fields are rejected, empty captions and
missing files produce indexed diagnostics, and duplicate video/caption pairs
are rejected. `caption_prefix` applies trigger tokens before cache identity is
computed.

`assign_video_bucket` chooses the closest configured aspect ratio, calculates
an aspect-preserving resize and centered crop, resamples the available frame
count to the target FPS, and selects the largest compatible temporal bucket.
Wan-style frame buckets default to the form `4n+1`.

## Cache identity

`preprocess_cache_key` hashes:

- normalized source path, size, and modification time;
- transformed caption;
- model family and transformer/text/image/VAE checkpoint identities;
- preprocessing version;
- output resolution, frames, FPS, resize, and crop;
- VAE scale and tensor dtype.

Wan entries are atomically written SafeTensors files containing rank-4 latents
and rank-2 text context. Wan I2V entries additionally contain the official
20-channel first-frame VAE/mask conditioning and 1,280-channel CLIP vision
tokens. The reference image is the first frame of the same manifest video, so
the manifest schema remains identical for T2V and I2V. LTX entries contain
rank-2 patched latents, rank-2
Gemma/connector text context, and rank-2 time/height/width positions. Cache
inspection validates the family format, identity, tensor inventory, ranks, and
matching LTX latent/position token counts before loading. A changed identity
becomes a different path, so interrupted preprocessing can safely skip valid
entries and regenerate only stale or missing ones.

`CachedBatchProvider` and `LTXCachedBatchProvider` sample entries
deterministically from the training RNG, reject mixed shapes within a batch,
stack canonical batch dimensions, and optionally transfer model inputs to
CUDA. Because sampling consumes the checkpointed job RNG, resumed training
reproduces the same subsequent batches.
