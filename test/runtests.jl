using GradPower
using Test
using SparseArrays

include("test_case9data.jl")

@testset "static" begin
    @testset "network" begin
        include("test_network.jl")
    end
    @testset "power flow" begin
        include("test_pflow.jl")
    end
end
