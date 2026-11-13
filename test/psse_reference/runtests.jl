# PSS/E reference tests. Every `test_*.jl` in this directory is one device model
# validated against PSS/E 35; see README.md for the data and how to regenerate it.
#
#   julia --project=test test/psse_reference/runtests.jl     # these tests alone
using Test

isdefined(@__MODULE__, :PSSEReference) || include(joinpath(@__DIR__, "harness.jl"))
using .PSSEReference

@testset "PSS/E reference" begin
    include(joinpath(@__DIR__, "selftest.jl"))
    for f in sort(readdir(@__DIR__))
        startswith(f, "test_") && endswith(f, ".jl") && include(joinpath(@__DIR__, f))
    end
end
