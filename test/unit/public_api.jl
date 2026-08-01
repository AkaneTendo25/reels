@testset "exported public API has docstrings" begin
    undocumented = Symbol[]
    for name in names(Reels)
        binding = Base.Docs.Binding(Reels, name)
        Base.Docs.doc(binding) === nothing && push!(undocumented, name)
    end
    @test isempty(sort!(undocumented))
end
