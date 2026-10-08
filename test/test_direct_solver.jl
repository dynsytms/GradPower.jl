# Sparse direct solver backends on the CPU: SparseDirectSolver.jl must
# reproduce the KLU trajectories on every CPU Newton path.

using LinearAlgebra, SparseArrays

const EX_DS = joinpath(@__DIR__, "..", "examples")

function _ds_case(raw, dyr; fault::Bool = true)
    ps = from_psse(joinpath(EX_DS, raw), joinpath(EX_DS, dyr))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    fault && GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    return ps, dp
end

const DS_CASES = [
    ("2bus",      "2bus.raw",      "2bus.dyr"),
    ("ieee9 gov", "ieee9_v33.raw", "ieee9bus_gov.dyr"),
]

@testset "direct_solver" begin

@testset "backend selection without extensions" begin
    A = sparse([4.0 1.0; 1.0 3.0])
    @test GradPower.ds_solver(KLUBackend(), A) isa GradPower.KLU.KLUFactorization
    @test_throws ArgumentError GradPower.ds_solver(KLUBackend(), [1, 2, 3], [1, 2], [1.0, 2.0], "G", 'F')
    if Base.get_extension(GradPower, :GradPowerSparseDirectSolverExt) === nothing
        @test_throws ErrorException GradPower.ds_solver(SparseDirectSolverBackend(), A)
    end
    if Base.get_extension(GradPower, :GradPowerCUDSSExt) === nothing
        @test_throws ErrorException GradPower.ds_solver(CUDSSBackend(), A)
    end
end

@eval using SparseDirectSolver

@testset "SparseDirectSolver factorization API" begin
    @test Base.get_extension(GradPower, :GradPowerSparseDirectSolverExt) !== nothing
    n = 50
    A = sprandn(n, n, 0.1) + 10I
    b = randn(n)
    F = GradPower.ds_solver(SparseDirectSolverBackend(), A)
    x = similar(b)
    ldiv!(x, F, b)
    @test norm(A * x - b) / norm(b) < 1e-10
    # Refactorization with new values on the same pattern
    A2 = copy(A); nonzeros(A2) .*= 2.0
    F = GradPower.ds_factorize!(F, A2)
    ldiv!(x, F, b)
    @test norm(A2 * x - b) / norm(b) < 1e-10
end

@testset "integrate! solver=$solver: $label" for (label, raw, dyr) in DS_CASES,
                                                 solver in (:monolithic, :schur, :schur_gmres)
    ps, dp = _ds_case(raw, dyr)
    z0 = copy(dp.zvec)
    _, traj_klu = GradPower.integrate!(dp, ps, 0.5; dt=1.0/120.0, solver, linear_solver=KLUBackend())
    dp.zvec .= z0
    _, traj_sds = GradPower.integrate!(dp, ps, 0.5; dt=1.0/120.0, solver,
                                      linear_solver=SparseDirectSolverBackend())
    @test maximum(abs, traj_sds .- traj_klu) < 1e-8
end

@testset "integrate_batched! M=$M: $label" for (label, raw, dyr) in DS_CASES, M in (1, 4)
    ps, dp = _ds_case(raw, dyr)
    bl = GradPower.BatchedLayout(dp, ps, M)
    _, trajs_klu = GradPower.integrate_batched!(bl, ps, 0.5; dt=1.0/120.0)
    bl = GradPower.BatchedLayout(dp, ps, M)
    _, trajs_sds = GradPower.integrate_batched!(bl, ps, 0.5; dt=1.0/120.0,
                                               linear_solver=SparseDirectSolverBackend())
    for m in 1:M
        @test maximum(abs, trajs_sds[m] .- trajs_klu[m]) < 1e-8
    end
end

end
