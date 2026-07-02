using GradPower
using Test
using SparseArrays
using LinearAlgebra
using FiniteDiff

include("test_case9data.jl")

@testset "static" begin
    @testset "network" begin
        include("test_network.jl")
    end
    @testset "power flow" begin
        include("test_pflow.jl")
    end
    include("test_matpower_parser.jl")
end
@testset "dynamic" begin
    include("test_integrate.jl")
end
@testset "layout" begin
    include("test_layout.jl")
end
@testset "clusters" begin
    include("test_clusters.jl")
end
@testset "kernels" begin
    include("test_genrou_kernel.jl")
    include("test_ieesgo_kernel.jl")
    include("test_zipload_kernel.jl")
    include("test_kernel_allocations.jl")
    include("test_device_kernels.jl")
    include("test_fd_jacobian.jl")
end
@testset "ka_wrappers" begin
    include("test_ka_wrappers.jl")
end
@testset "batched_layout" begin
    include("test_batched_layout.jl")
end
@testset "gpu_backend" begin
    include("test_gpu_backend.jl")
end
@testset "coupling" begin
    include("test_coupling.jl")
end
@testset "lockstep" begin
    include("test_lockstep.jl")
end
