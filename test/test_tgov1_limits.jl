using Test
using GradPower
using FiniteDiff

function _limited_tgov1_setup(; side=:upper)
    ex = joinpath(@__DIR__, "..", "examples")
    ps = from_psse(joinpath(ex, "2bus.raw"), joinpath(ex, "2bus_TGOV1.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for device in ps.dynamic.devices
        device.dtype isa GradPower.ZIPLoad && (device.dtype.α = 0.5)
    end
    dp = DynamicProblem(ps)
    initialize_dynamics!(dp, ps)
    table = ps.dynamic.layout.tgov1
    pp = Int(table.par_ptr[1])
    state = Int(table.diff_ptr[1]) + 1
    initial = dp.zvec[state]
    dp.pvec[pp + 2] = initial + 0.01
    dp.pvec[pp + 3] = initial - 0.01
    dp.pvec[pp + 7] += side === :upper ? 0.05 : -0.05
    return ps, dp, state, initial
end

@testset "TGOV1 fixed-state limits" begin
    for side in (:upper, :lower), method in (:active_set, :complementarity)
        ps, dp, state, initial = _limited_tgov1_setup(side=side)
        events = LimitEvent[]
        _, trajectory = integrate!(dp, ps, 0.05; dt=0.005,
            limit_method=method, limit_events=events)
        bound = initial + (side === :upper ? 0.01 : -0.01)
        values = trajectory[state, :]
        @test minimum(values) >= initial - 0.01 - 1e-10
        @test maximum(values) <= initial + 0.01 + 1e-10
        @test any(isapprox(bound; atol=1e-9), values)
        @test any(event -> event.action == :activate && event.side == side, events)
    end

    ps_none, dp_none, _, _ = _limited_tgov1_setup()
    ps_default, dp_default, _, _ = _limited_tgov1_setup()
    t_none, z_none = integrate!(dp_none, ps_none, 0.05; dt=0.005, limit_method=:none)
    t_default, z_default = integrate!(dp_default, ps_default, 0.05; dt=0.005)
    @test t_none == t_default
    @test z_none == z_default

    ps, dp, _, _ = _limited_tgov1_setup()
    @test_throws ArgumentError integrate!(dp, ps, 0.05; limit_method=:active_set, solver=:schur)

    ps, dp, state, initial = _limited_tgov1_setup()
    J = GradPower.preallocate_jacobian(ps)
    limits = GradPower.build_limit_workspace(ps, J, dp.zvec, dp.pvec;
        method=:complementarity, mu=0.0, tolerance=1e-10)
    zold = copy(dp.zvec)
    z = copy(dp.zvec)
    z[state] = initial + 0.005
    f = zeros(length(z))
    GradPower.beuler_batched!(f, z, zold, dp.uvec, dp.pvec, ps.dynamic,
        ps.network.ybus_real, ps.dynamic.layout, ps.dynamic.diff_dim, 0.005,
        nothing, limits)
    GradPower.beuler_jac_batched!(J, z, dp.uvec, dp.pvec, ps.dynamic,
        ps.network.ybus_real, ps.dynamic.layout, ps.dynamic.diff_dim, 0.005,
        zold, limits)
    Jfd = FiniteDiff.finite_difference_jacobian(z) do zz
        ff = zeros(length(zz))
        GradPower.beuler_batched!(ff, zz, zold, dp.uvec, dp.pvec, ps.dynamic,
            ps.network.ybus_real, ps.dynamic.layout, ps.dynamic.diff_dim, 0.005,
            nothing, limits)
        ff
    end
    @test maximum(abs, J[state, :] .- Jfd[state, :]) < 1e-6

    for mu in (0.0, 1e-6, 1e-4, 1e-2)
        ps_mu, dp_mu, _, _ = _limited_tgov1_setup()
        _, trajectory = integrate!(dp_mu, ps_mu, 0.02; dt=0.005,
            limit_method=:complementarity, limit_mu=mu)
        @test all(isfinite, trajectory)
    end

    ps_event, dp_event, state, initial = _limited_tgov1_setup()
    J_event = GradPower.preallocate_jacobian(ps_event)
    events = LimitEvent[]
    workspace = GradPower.build_limit_workspace(ps_event, J_event, dp_event.zvec,
        dp_event.pvec; method=:active_set, mu=0.0, tolerance=1e-10, events=events)
    descriptor = only(workspace.descriptors)
    upper = dp_event.pvec[descriptor.upper_parameter_index]
    zold = copy(dp_event.zvec)
    z = copy(zold)
    z[state] = upper
    workspace.modes[1] = GradPower.LIMIT_UPPER
    pp = descriptor.upper_parameter_index - 2
    dp_event.pvec[pp + 7] -= 0.1
    GradPower.record_limit_events!(workspace, z, zold, dp_event.pvec, 0.005, 0.005)
    @test workspace.modes[1] == GradPower.LIMIT_FREE
    @test only(events).action == :release

    f = zeros(length(z))
    GradPower.beuler_batched!(f, z, zold, dp_event.uvec, dp_event.pvec,
        ps_event.dynamic, ps_event.network.ybus_real, ps_event.dynamic.layout,
        ps_event.dynamic.diff_dim, 0.0, nothing, workspace)
    @test f[state] == z[state] - zold[state]

    GradPower.beuler_batched!(f, z, zold, dp_event.uvec, dp_event.pvec,
        ps_event.dynamic, ps_event.network.ybus_real, ps_event.dynamic.layout,
        ps_event.dynamic.diff_dim, 0.005, nothing, workspace)
    GradPower.beuler_jac_batched!(J_event, z, dp_event.uvec, dp_event.pvec,
        ps_event.dynamic, ps_event.network.ybus_real, ps_event.dynamic.layout,
        ps_event.dynamic.diff_dim, 0.005, zold, workspace)
    function allocation_check(f, J, z, zold, p, workspace)
        residual_bytes = @allocated GradPower.apply_limit_residual!(
            f, z, zold, p, 0.005, workspace)
        jacobian_bytes = @allocated GradPower.apply_limit_jacobian!(
            J, z, zold, p, 0.005, workspace)
        return residual_bytes, jacobian_bytes
    end
    allocation_check(f, J_event, z, zold, dp_event.pvec, workspace)
    @test allocation_check(f, J_event, z, zold, dp_event.pvec, workspace) == (0, 0)
end
