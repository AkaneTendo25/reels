@testset "SafeTensors streaming I/O" begin
    mktempdir() do dir
        path = joinpath(dir, "adapter.safetensors")
        tensors = Dict("z" => reshape(Float32.(1:6), 2, 3), "a" => Float16[1, 2])
        write_safetensors(path, tensors; metadata=Dict("format" => "reels"))
        header = inspect_safetensors(path)
        @test header.metadata["format"] == "reels"
        @test sort(collect(keys(header.tensors))) == ["a", "z"]
        @test load_safetensor(path, "z") == permutedims(tensors["z"])
        @test load_safetensor(path, "a") == tensors["a"]
        @test_throws KeyError load_safetensor(path, "missing")
    end
end

@testset "complete JSON scalar and Unicode support" begin
    value = Reels.parse_json(
        """{"enabled":true,"disabled":false,"missing":null,"scale":1.25,"emoji":"\\ud83d\\ude80"}""")
    @test value["enabled"] === true
    @test value["disabled"] === false
    @test value["missing"] === nothing
    @test value["scale"] == 1.25
    @test value["emoji"] == "🚀"
    encoded = Reels.json_encode(Dict(
        "enabled" => true, "missing" => nothing,
        "scale" => 1.25, "shape" => (1, 2)))
    @test Reels.parse_json(encoded) == Dict{String,Any}(
        "enabled" => true, "missing" => nothing,
        "scale" => 1.25, "shape" => Any[1, 2])
end
