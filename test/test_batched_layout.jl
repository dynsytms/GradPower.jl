# Unit tests for BatchedLayout (phase 14a D4).

@testset "BatchedLayout constructor" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    M = 4
    bl = GradPower.BatchedLayout(dp, ps, M)

    @test bl.M == M
    @test bl.sys_dim == length(dp.zvec)
    @test size(bl.z) == (M, bl.sys_dim)
    @test size(bl.p) == (M, length(dp.pvec))
    @test size(bl.u) == (M, length(dp.uvec))
    @test size(bl.f) == (M, bl.sys_dim)

    # Each scenario matches single-scenario IC
    for m in 1:M
        @test bl.z[m, :] == dp.zvec
        @test bl.p[m, :] == dp.pvec
        @test bl.u[m, :] == dp.uvec
    end
end

@testset "uvec_idx routing" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    M = 3
    bl = GradPower.BatchedLayout(dp, ps, M)

    # Apply uvec routing (same as done inside residual evaluation)
    GradPower._apply_uvec_routing_batched!(bl.u, bl.z, bl.uvec_idx, M)

    for j in eachindex(bl.uvec_idx)
        if bl.uvec_idx[j] != 0
            for m in 1:M
                @test bl.u[m, j] == bl.z[m, bl.uvec_idx[j]]
            end
        end
    end
end

@testset "Batched residual matches single-scenario" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout

    # Single-scenario reference
    f_ref = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    # Batched
    M = 3
    bl = GradPower.BatchedLayout(dp, ps, M)
    GradPower._rhs_fun_all_scenarios!(bl, dyn, L)

    for m in 1:M
        @test maximum(abs, bl.f[m, :] - f_ref) <= 1e-14
    end
end

@testset "Batched Jacobian nzval matches single-scenario" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout

    # Single-scenario reference
    J_ref = GradPower.preallocate_jacobian(ps)
    fill!(J_ref.nzval, 0.0)
    GradPower._rhs_jac_batched!(J_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    # Batched
    M = 3
    bl = GradPower.BatchedLayout(dp, ps, M)
    GradPower._rhs_jac_all_scenarios!(bl, dyn, L)

    for m in 1:M
        @test maximum(abs, bl.J_nzval[m, :] - J_ref.nzval) <= 1e-14
    end
end
