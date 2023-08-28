using GradPower
using Test
using SparseArrays

include("test_case9data.jl")

@testset "utils" begin
    include("test_utils.jl")
end

@testset "static" begin
    @testset "network" begin
        include("test_network.jl")
    end
    @testset "power flow" begin
        include("test_pflow.jl")
    end
end
