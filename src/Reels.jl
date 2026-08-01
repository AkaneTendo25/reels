module Reels

using Dates
using BFloat16s: BFloat16
import CUDA
import cuDNN
import ChainRulesCore
using Libdl
using LinearAlgebra
import NNlib
import NCCL_jll
using Random
using SHA
using Sockets
import SpecialFunctions
using TOML
import Zygote
include("models/interface.jl")
include("config/schema.jl")
include("config/load.jl")
include("checkpoints/json.jl")
include("checkpoints/safetensors.jl")
include("checkpoints/tensor_layout.jl")
include("checkpoints/state_dict.jl")
include("layers/quantization.jl")
include("layers/lora.jl")
include("layers/linear.jl")
include("layers/normalization.jl")
include("layers/rope.jl")
include("layers/modulation.jl")
include("layers/attention.jl")
include("objectives/flow_matching.jl")
include("training/precision.jl")
include("training/optimizer.jl")
include("training/state.jl")
include("training/metrics.jl")
include("training/distributed.jl")
include("training/runner.jl")
include("models/synthetic.jl")
include("models/wan21/config.jl")
include("models/wan21/tokenizer.jl")
include("models/wan21/mappings.jl")
include("models/wan21/text_encoder.jl")
include("models/wan21/transformer.jl")
include("models/wan21/vae.jl")
include("models/wan21/lora.jl")
include("models/wan21/backend.jl")
include("models/wan21/image_encoder.jl")
include("models/ltx23/config.jl")
include("models/ltx23/mappings.jl")
include("models/ltx23/patchifier.jl")
include("models/ltx23/vae.jl")
include("models/ltx23/transformer.jl")
include("models/ltx23/gemma.jl")
include("models/ltx23/text_connector.jl")
include("models/ltx23/lora.jl")
include("models/ltx23/backend.jl")
include("training/validation.jl")
include("training/ad_rules.jl")
include("training/wan.jl")
include("training/ltx23.jl")
include("training/ltx23_runner.jl")
include("training/wan_runner.jl")
include("data/manifest.jl")
include("data/buckets.jl")
include("data/media.jl")
include("data/cache.jl")
include("data/provider.jl")
include("cli/main.jl")

export AbstractVideoModelBackend, model_family, load_config, load_transformer,
    load_text_encoder, load_vae, load_image_encoder, encode_text,
    encode_image, encode_video,
    decode_video,
    prepare_conditioning,
    sample_training_target, predict_velocity, lora_targets, export_mapping
export precision_eltype, array_transfer
export ModelConfig, DataConfig, LoRAConfig, TrainingConfig, SchedulerConfig,
    CheckpointConfig, ValidationConfig, FlowMatchingConfig, ReelsConfig,
    DistributedConfig, LowVRAMConfig, load_config_file, validate
export SafeTensorHeader, TensorInfo, inspect_safetensors, load_safetensor,
    write_safetensors, AbstractTensorSource, SingleTensorSource,
    ShardedTensorSource, open_tensor_source, tensor_keys, write_sharded_safetensors
export BFloat16
export TensorLayout, restore_source_layout, TensorSpec, StateDictAudit,
    audit_state_dict, load_state_tensor, SCALAR_LAYOUT, VECTOR_LAYOUT,
    LINEAR_OUT_IN, CONV_OUT_IN_SPATIAL, ROW_MAJOR_SOURCE, NATIVE_LAYOUT
export LoRALinear, lora_parameters, merged_weight, set_adapter_enabled!, lora_forward,
    lora_backward
export QuantizedMatrix, CPUOffloadedMatrix, quantize_frozen_matrix,
    FrozenWeightTransfer, frozen_weight_transfer
export quantized_umt5_state_dict, quantized_umt5_encoder_specs,
    write_quantized_umt5, load_quantized_umt5_encoder
export DenseLayer, linear, layernorm, gelu_tanh
export RMSNorm, rmsnorm, rmsnorm_backward
export RotaryEmbedding, apply_rope, rope_backward
export modulation, modulation_backward
export AttentionCache, reference_attention, attention_backward
export TiledAttentionCache, memory_efficient_attention,
    memory_efficient_attention_backward
export FlowSample, flow_sample, uniform_timestep, ltx_timestep_shift,
    shifted_logit_normal_timestep, sample_flow_timesteps
export flow_euler_timesteps, ltx23_sigma_schedule, ltx23_validation_sample,
    wan_validation_sample,
    ltx23_validation_comparison, wan_validation_comparison,
    WanValidationBatch, LTXValidationBatch, run_validation!
export AdamW, AdamWState, update!, learning_rate, weight_decay_mask
export TrainingState, save_checkpoint, load_checkpoint
export AbstractDistributedRuntime, SingleProcessRuntime, NCCLRuntime,
    init_distributed, close_distributed!, distributed_rank,
    distributed_world_size, is_main_process, allreduce_mean!,
    allreduce_gradients!, distributed_mean_scalar, distributed_barrier!
export TrainingJob, train!
export SyntheticBackend, SyntheticModel, synthetic_batch
export Wan21, Wan21Config, wan21_config, wan21_transformer_specs,
    WanTextConditioner, WanTextEncoding,
    SentencePieceTokenizer, TokenizedText, sentencepiece_library,
    clean_wan_caption, sentencepiece_ids, tokenize_wan, tokenize_gemma,
    UMT5Config, UMT5Encoder, T5Attention, T5FeedForward, T5EncoderBlock,
    umt5_encoder_specs, t5_relative_buckets, t5_position_bias, t5_rmsnorm,
    t5_attention, t5_block_forward, umt5_forward, load_umt5_encoder,
    umt5_encoder_state_dict,
    VAEConv3D, VAERMSNorm, VAEResidualBlock, VAEAttentionBlock, VAEDownsample,
    VAEUpsample, WanVAEConfig, WanVAEEncoder, WanVAEDecoder, WanVAE,
    vae_conv3d, vae_rmsnorm, vae_silu, vae_residual_forward,
    vae_attention_forward, vae_downsample_forward, wan_vae_encoder_forward,
    vae_upsample_forward, wan_vae_decoder_forward,
    wan_vae_encoder_specs, wan_vae_decoder_specs, wan_vae_specs,
    load_wan_vae_encoder, load_wan_vae_decoder,
    load_wan_vae,
    wan_vae_encoder_state_dict, wan_vae_decoder_state_dict,
    WAN_VAE_MEAN, WAN_VAE_STD,
    scale_wan_latents, unscale_wan_latents,
    prepare_wan_i2v_conditioning,
    WanCLIPVisionConfig, WanCLIPLayerNorm, WanCLIPAttention, WanCLIPMLP,
    WanCLIPEncoderLayer, WanCLIPVisionEncoder, wan_clip_vision_specs,
    wan_clip_attention_forward, wan_clip_layer_forward,
    wan_clip_vision_forward, load_wan_clip_vision,
    wan_clip_vision_state_dict,
    preprocess_wan_clip_image, WAN_CLIP_MEAN, WAN_CLIP_STD,
    WanAttention, WanTransformerBlock, wan_rope3d, wan_block_forward,
    load_wan_block, WanTransformer, patchify, unpatchify,
    sinusoidal_embedding, wan_transformer_forward, load_wan_transformer,
    wan_transformer_state_dict, move_to_device, inject_wan_lora,
    wan_lora_layers, wan_lora_parameters, wan_lora_state_dict,
    save_wan_lora, load_wan_lora!, wan_lora_loss_and_gradients,
    wan_lora_step!, WanLatentBatch, WanTrainingJob
export LTX23, LTX23Config, ltx23_config, ltx23_video_dim, ltx23_audio_dim,
    ltx23_adaln_parameters, ltx23_transformer_specs,
    LTX23_VIDEO_SCALE_FACTORS, ltx23_patchify_latents,
    ltx23_unpatchify_latents, ltx23_patch_bounds, ltx23_pixel_bounds,
    ltx23_patch_positions,
    LTXVAEBlockConfig, LTXVideoVAEConfig, LTX23_VAE_DEFAULT_BLOCKS,
    ltx23_vae_config, ltx23_vae_encoder_specs, LTXVAEResidualBlock,
    LTXVAEDownsample, LTXVideoVAEEncoder, ltx23_vae_patchify,
    ltx23_pixel_norm, ltx23_causal_conv3d, ltx23_vae_residual_forward,
    ltx23_vae_downsample_forward, ltx23_vae_encoder_forward,
    load_ltx23_vae_encoder,
    Gemma3TextConfig, gemma3_text_config, gemma3_text_encoder_specs,
    Gemma3Norm, Gemma3Attention, Gemma3MLP, Gemma3DecoderBlock,
    Gemma3TextEncoder, gemma3_norm, gemma3_attention_forward,
    gemma3_block_forward, gemma3_forward, load_gemma3_text_encoder,
    LTXTextConnectorConfig, ltx23_text_connector_config,
    ltx23_text_connector_specs, LTXTextConnectorBlock, LTXTextConnector,
    LTXTextConditioner, LTXTextEncoding,
    ltx23_gemma_features, ltx23_text_connector_forward,
    ltx23_text_conditioner_forward,
    load_ltx23_text_connector,
    LTXTimestepEmbedder, LTXAdaLN, LTXAttention, LTXVideoBlock,
    LTXVideoTransformer, ltx_timestep_embedding, ltx_adaln_forward,
    ltx_rope_frequencies, ltx_apply_rope, ltx_attention_forward,
    ltx_video_block_forward, ltx_video_transformer_forward,
    load_ltx23_video_transformer, inject_ltx23_lora, ltx23_lora_layers,
    ltx23_lora_parameters, ltx23_lora_state_dict, save_ltx23_lora,
    load_ltx23_lora!, ltx23_lora_loss_and_gradients, ltx23_lora_step!
export LTXLatentBatch, LTXTrainingJob
export VideoSample, load_video_manifest, VideoMetadata, BucketAssignment,
    assign_video_bucket, PreprocessIdentity, source_media_fingerprint,
    checkpoint_fingerprint,
    preprocess_cache_key, cache_entry_path, write_preprocess_cache,
    write_wan_preprocess_cache, build_wan_preprocess_cache,
    write_ltx23_preprocess_cache, inspect_ltx23_preprocess_cache,
    load_ltx23_preprocess_cache, ltx23_cache_is_valid,
    build_ltx23_preprocess_cache,
    inspect_preprocess_cache, load_preprocess_cache, cache_is_valid,
    inspect_validation_cache, load_validation_cache,
    build_wan_validation_caches, build_ltx23_validation_caches,
    CachedBatchProvider, LTXCachedBatchProvider,
    distributed_sample_indices,
    ffmpeg_executable, ffprobe_executable, probe_video,
    decode_video_frames, decode_video_sample, write_video_frames
export reels_main, cli_preprocess, cli_train, cli_inspect, cli_validate_run

end
