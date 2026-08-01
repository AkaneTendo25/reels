@testset "tensor layouts and state dict audit" begin
    mktempdir() do dir
        path = joinpath(dir, "weights.safetensors")
        linear = reshape(Float32.(1:6), 2, 3)
        write_safetensors(path, Dict("linear.weight" => linear,
                                    "extra" => Float32[1]))
        header = inspect_safetensors(path)
        spec = TensorSpec("linear.weight", [2, 3], LINEAR_OUT_IN)
        audit = audit_state_dict(header, [spec])
        @test audit.missing == String[]
        @test audit.unexpected == ["extra"]
        @test isempty(audit.shape_mismatches)
        @test load_state_tensor(path, spec) == linear
        bad = TensorSpec("linear.weight", [3, 2], LINEAR_OUT_IN)
        @test length(audit_state_dict(header, [bad]; allow_unexpected=true).shape_mismatches) == 1
    end
end

@testset "sharded SafeTensors source" begin
    mktempdir() do dir
        index = joinpath(dir, "model.safetensors.index.json")
        shards = Dict(
            "model-00001-of-00002.safetensors" =>
                Dict("a" => reshape(Float32.(1:6), 2, 3)),
            "model-00002-of-00002.safetensors" =>
                Dict("b" => Float32[7, 8]),
        )
        write_sharded_safetensors(index, shards)
        source = open_tensor_source(index)
        @test source isa ShardedTensorSource
        @test sort(tensor_keys(source)) == ["a", "b"]
        @test load_state_tensor(source,
            TensorSpec("a", [2, 3], LINEAR_OUT_IN)) == shards[
                "model-00001-of-00002.safetensors"]["a"]
        specs = [TensorSpec("a", [2, 3], LINEAR_OUT_IN),
                 TensorSpec("b", [2], VECTOR_LAYOUT)]
        @test isempty(audit_state_dict(source, specs))
        @test_throws KeyError load_safetensor(source, "missing")
    end
end

@testset "SafeTensors checkpoint directory resolution" begin
    mktempdir() do directory
        index = joinpath(directory, "model.safetensors.index.json")
        write_sharded_safetensors(index, Dict(
            "model-00001-of-00002.safetensors" =>
                Dict("left" => reshape(Float32[1, 2], 2, 1)),
            "model-00002-of-00002.safetensors" =>
                Dict("right" => Float32[3, 4]),
        ))
        source = open_tensor_source(directory)
        @test source isa Reels.ShardedTensorSource
        @test sort(tensor_keys(source)) == ["left", "right"]
    end

    mktempdir() do directory
        file = joinpath(directory, "model.safetensors")
        write_safetensors(file, Dict("value" => Float32[1]))
        @test tensor_keys(open_tensor_source(directory)) == ["value"]
        write_safetensors(
            joinpath(directory, "other.safetensors"),
            Dict("other" => Float32[2]))
        @test_throws ArgumentError open_tensor_source(directory)
    end

    mktempdir() do directory
        @test_throws ArgumentError open_tensor_source(directory)
    end
end
