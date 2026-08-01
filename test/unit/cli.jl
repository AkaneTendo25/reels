@testset "Reels CLI inspection and run validation" begin
    mktempdir() do directory
        tensor_path = joinpath(directory, "tensor.safetensors")
        write_safetensors(tensor_path, Dict("value" => Float32[1, 2, 3]);
            metadata=Dict("kind" => "test"))
        inspected = cli_inspect(tensor_path)
        @test haskey(inspected.tensors, "value")
        @test reels_main(["inspect", tensor_path]).metadata["kind"] == "test"

        config = Wan21Config(patch_size=(1, 2, 2), text_length=2,
            input_channels=1, hidden_size=6, ffn_size=8, frequency_size=4,
            text_size=5, output_channels=1, heads=1, layers=1)
        model = inject_wan_lora(WanTransformer(config);
            targets=["blocks.0.self_attn.q.weight"], rank=1)
        adapter = joinpath(directory, "adapter-final.safetensors")
        save_wan_lora(adapter, model; base_model="tiny")
        validated = cli_validate_run(directory)
        @test validated.adapter == adapter
        @test isempty(validated.checkpoints)
        @test reels_main(["validate", "--run", directory]).adapter == adapter
    end
end

@testset "LTX run validation" begin
    mktempdir() do directory
        config = LTX23Config(video_only=true, video_heads=1,
            video_head_dim=6, video_channels=3, video_context_dim=4,
            layers=1)
        model = inject_ltx23_lora(LTXVideoTransformer(config);
            targets=["transformer_blocks.0.attn1.to_q.weight"], rank=1)
        adapter = joinpath(directory, "adapter-final.safetensors")
        save_ltx23_lora(adapter, model; base_model="tiny-ltx")
        validated = cli_validate_run(directory)
        @test validated.adapter == adapter
        @test inspect_safetensors(adapter).metadata["model_family"] == "ltx23"
    end
end
