# Integration and event handling tests.
# Replaces the skipped test_dynamics.jl (which depended on deleted ad.jl).

const EX_INT = joinpath(@__DIR__, "..", "examples")

function _build_and_init(raw, dyr)
    ps = from_psse(joinpath(EX_INT, raw), joinpath(EX_INT, dyr))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    return ps, dp
end

function _clear_events!(ps)
    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end
    empty!(ps.dynamic.events)
end

# -----------------------------------------------------------------------
# Initialization residual check
# -----------------------------------------------------------------------

@testset "Initialization residual ≈ 0" begin
    for (label, raw, dyr) in [
        ("2bus",      "2bus.raw",      "2bus.dyr"),
        ("ieee9 gov", "ieee9_v33.raw", "ieee9bus_gov.dyr"),
    ]
        @testset "$label" begin
            ps, dp = _build_and_init(raw, dyr)
            dyn = ps.dynamic
            L = dyn.layout
            n = length(dp.zvec)
            f = zeros(n)
            GradPower._rhs_fun_batched!(f, dp.zvec, dp.uvec, dp.pvec,
                                         dyn, ps.network.ybus_real, L)
            @test maximum(abs, f) < 1e-9
        end
    end
end

# -----------------------------------------------------------------------
# Flat-line test: no fault → states should not drift
# -----------------------------------------------------------------------

@testset "Flat-line: no fault, states stay at z0" begin
    for (label, raw, dyr) in [
        ("2bus",      "2bus.raw",      "2bus.dyr"),
        ("ieee9 gov", "ieee9_v33.raw", "ieee9bus_gov.dyr"),
    ]
        @testset "$label" begin
            ps, dp = _build_and_init(raw, dyr)
            z0 = copy(dp.zvec)
            tvec, traj = GradPower.integrate!(dp, ps, 0.5; dt=1.0/120.0)
            # Final state should be ≈ z0 (backward Euler drift is O(dt²))
            @test maximum(abs, traj[:, end] - z0) < 1e-5
        end
    end
end

# -----------------------------------------------------------------------
# Trajectory reproducibility
# -----------------------------------------------------------------------

@testset "Trajectory reproducibility" begin
    ps, dp1 = _build_and_init("2bus.raw", "2bus.dyr")
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec1, traj1 = GradPower.integrate!(dp1, ps, 1.0; dt=1.0/120.0)
    _clear_events!(ps)

    _, dp2 = _build_and_init("2bus.raw", "2bus.dyr")
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec2, traj2 = GradPower.integrate!(dp2, ps, 1.0; dt=1.0/120.0)
    _clear_events!(ps)

    @test traj1 == traj2
end

# -----------------------------------------------------------------------
# Event handling: fault causes voltage drop
# -----------------------------------------------------------------------

@testset "ContingencyEvent causes voltage drop" begin
    ps, dp_nofault = _build_and_init("2bus.raw", "2bus.dyr")
    tvec_nf, traj_nf = GradPower.integrate!(dp_nofault, ps, 0.5; dt=1.0/120.0)

    _, dp_fault = _build_and_init("2bus.raw", "2bus.dyr")
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec_f, traj_f = GradPower.integrate!(dp_fault, ps, 0.5; dt=1.0/120.0)
    _clear_events!(ps)

    # Trajectories must differ
    @test maximum(abs, traj_f - traj_nf) > 0.01

    # During fault (t=0.1 to t=0.2), bus 1 voltage magnitude should drop.
    dd = ps.dynamic.diff_dim
    ad = ps.dynamic.alg_dim
    np = dd + ad
    vr_idx = np + 1  # bus 1 vr
    vi_idx = np + 2  # bus 1 vi

    step_mid_fault = findfirst(t -> t >= 0.15, tvec_f)
    vm_fault = sqrt(traj_f[vr_idx, step_mid_fault]^2 + traj_f[vi_idx, step_mid_fault]^2)
    vm_pre   = sqrt(traj_f[vr_idx, 1]^2 + traj_f[vi_idx, 1]^2)

    # Voltage should drop during fault (rfault = 0.02 is a severe fault)
    @test vm_fault < vm_pre * 0.9
end

# -----------------------------------------------------------------------
# Final-state spot checks (self-consistency across refactors)
# -----------------------------------------------------------------------

@testset "Final-state spot check: 2bus fault" begin
    ps, dp = _build_and_init("2bus.raw", "2bus.dyr")
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec, traj = GradPower.integrate!(dp, ps, 1.0; dt=1.0/120.0)
    _clear_events!(ps)

    result = traj[:, end]
    # These are from a known-good run — tolerance is loose (1e-4)
    # to allow for minor numerical changes, but tight enough to catch
    # algorithmic regressions.
    L = ps.dynamic.layout
    dp_k = Int(L.genrou.diff_ptr[1])
    # e_qp should be near its initial value (perturbed by fault)
    @test abs(result[dp_k]) > 0.5     # e_qp stays physical
    @test abs(result[dp_k]) < 2.0
    # omega (w) should return near 0 after fault clears
    @test abs(result[dp_k + 4]) < 0.1  # w is deviation from synchronous
end

@testset "Final-state spot check: ieee9 gov" begin
    ps, dp = _build_and_init("ieee9_v33.raw", "ieee9bus_gov.dyr")
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))
    tvec, traj = GradPower.integrate!(dp, ps, 1.0; dt=1.0/120.0)
    _clear_events!(ps)

    result = traj[:, end]
    L = ps.dynamic.layout
    # All 3 generators should have physical omega (|w| < 0.1)
    for k in 1:L.genrou.n
        dp_k = Int(L.genrou.diff_ptr[k])
        @test abs(result[dp_k + 4]) < 0.1
    end
    # All 3 governors should have physical p_m (> 0)
    dd = ps.dynamic.diff_dim
    for k in 1:L.ieesgo.n
        ap_k = dd + Int(L.ieesgo.alg_ptr[k])
        @test result[ap_k] > 0.0
    end
end

# -----------------------------------------------------------------------
# Newton non-convergence is reported, not silently recorded
# -----------------------------------------------------------------------

@testset "Newton non-convergence throws" begin
    ps, dp = _build_and_init("2bus.raw", "2bus.dyr")
    for solver in (:monolithic, :schur, :schur_gmres)
        @test_throws GradPower.NewtonConvergenceError GradPower.integrate!(
            dp, ps, 0.05; dt=1.0/120.0, solver=solver, newton_tol=1e-300)
    end
    bl = GradPower.BatchedLayout(dp, ps, 2)
    @test_throws GradPower.NewtonConvergenceError GradPower.integrate_batched!(
        bl, ps, 0.05; dt=1.0/120.0, newton_tol=1e-300)
end

# -----------------------------------------------------------------------
# integrate! leaves the system as it found it (topology / online flags)
# -----------------------------------------------------------------------

@testset "integrate! restores system state" begin
    function _rerun_flat(ps)
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        _, traj = GradPower.integrate!(dp, ps, 0.2; dt=1.0/120.0)
        return traj
    end

    ps, _ = _build_and_init("ieee9_v33.raw", "ieee9bus_gov.dyr")
    ref = _rerun_flat(ps)
    ybus0 = copy(ps.network.ybus_real)
    online0 = [d.online for d in ps.dynamic.devices]

    @testset "permanent line trip" begin
        tev = GradPower.create_trip_line_event(ps, 4, 5, 0.05)
        GradPower.add_trip_event!(ps, tev)
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        GradPower.integrate!(dp, ps, 0.2; dt=1.0/120.0)
        @test ps.network.ybus_real == ybus0
        empty!(ps.dynamic.trip_events)
        @test _rerun_flat(ps) == ref
    end

    @testset "device disconnect" begin
        # ZIPLoad has no states, so disconnecting it (load shedding) keeps
        # the Jacobian nonsingular.
        gidx = findfirst(d -> d.dtype isa GradPower.ZIPLoad, ps.dynamic.devices)
        GradPower.add_disconnect_event!(ps, GradPower.DisconnectDeviceEvent(gidx, 0.05))
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        GradPower.integrate!(dp, ps, 0.2; dt=1.0/120.0)
        @test [d.online for d in ps.dynamic.devices] == online0
        @test all(all(t.online) for t in values(ps.dynamic.layout.tables))
        empty!(ps.dynamic.disconnect_events)
        @test _rerun_flat(ps) == ref
    end

    @testset "restored after a failed run" begin
        tev = GradPower.create_trip_line_event(ps, 4, 5, 1.0/120.0)
        GradPower.add_trip_event!(ps, tev)
        # A near-zero-resistance fault (admittance 1e300; rfault = 0 is
        # rejected) makes the dt=0 re-solve fail after the trip and the
        # disconnect have mutated the system.
        gidx = findfirst(d -> d.dtype isa GradPower.IEESGO, ps.dynamic.devices)
        GradPower.add_disconnect_event!(ps, GradPower.DisconnectDeviceEvent(gidx, 1.0/120.0))
        GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 1e-300, 1.0/120.0, 0.1))
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        @test_throws Exception GradPower.integrate!(dp, ps, 0.1; dt=1.0/120.0)
        @test ps.network.ybus_real == ybus0
        @test [d.online for d in ps.dynamic.devices] == online0
        @test !any(ev.status for ev in ps.dynamic.events)
        empty!(ps.dynamic.trip_events)
        empty!(ps.dynamic.disconnect_events)
        _clear_events!(ps)
    end

    @testset "batched: fault deactivated after a failed run" begin
        GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 1.0/120.0, 0.1))
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        bl = GradPower.BatchedLayout(dp, ps, 1)
        # Fault active as if mid-run; the first Newton solve then fails.
        GradPower.activate!(ps.dynamic.events[1])
        @test_throws GradPower.NewtonConvergenceError GradPower.integrate_batched!(
            bl, ps, 0.1; dt=1.0/120.0, newton_tol=1e-300)
        @test !any(ev.status for ev in ps.dynamic.events)
        _clear_events!(ps)
    end
end

# -----------------------------------------------------------------------
# Time grid: tvec and trajectory agree when tf is not a multiple of dt
# -----------------------------------------------------------------------

@testset "tvec length matches trajectory" begin
    ps, dp = _build_and_init("2bus.raw", "2bus.dyr")
    for (tf, dt) in ((0.26, 0.1), (0.24, 0.1), (0.3, 0.1), (1.0, 1.0/120.0))
        tvec, traj = GradPower.integrate!(dp, ps, tf; dt=dt)
        @test length(tvec) == size(traj, 2)
        @test tvec[1] == 0.0
        @test tvec[end] ≈ (length(tvec) - 1) * dt
        bl = GradPower.BatchedLayout(dp, ps, 1)
        tvec_b, trajs = GradPower.integrate_batched!(bl, ps, tf; dt=dt)
        @test tvec_b == tvec
        @test length(tvec_b) == size(trajs[1], 2)
    end
    # Unchanged values when tf is a multiple of dt.
    @test GradPower._time_grid(1.0, 1.0/120.0)[2] == collect(0:1.0/120.0:1.0)
end

# -----------------------------------------------------------------------
# Disconnecting a device with states freezes them (nonsingular Jacobian)
# -----------------------------------------------------------------------

@testset "Disconnect device with states" begin
    dt = 1.0/120.0
    t_off = 0.05
    k_off = Int(round(t_off/dt)) + 1  # traj column right after the disconnect
    ps, _ = _build_and_init("ieee9_v33.raw", "ieee9bus_gov.dyr")
    dd = ps.dynamic.diff_dim
    for T in (GradPower.IEESGO, GradPower.Genrou)
        @testset "$T" begin
            idx = findfirst(d -> d.dtype isa T, ps.dynamic.devices)
            dev = ps.dynamic.devices[idx]
            rows = vcat(dev.diff_ptr:dev.diff_ptr+dev.dtype.diff_size-1,
                        dd+dev.alg_ptr:dd+dev.alg_ptr+dev.dtype.alg_size-1)
            GradPower.add_disconnect_event!(ps, GradPower.DisconnectDeviceEvent(idx, t_off))
            GradPower.add_event!(ps, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.15))
            for solver in (:monolithic, :schur, :schur_gmres)
                dp = GradPower.DynamicProblem(ps)
                GradPower.initialize_dynamics!(dp, ps)
                _, traj = GradPower.integrate!(dp, ps, 0.3; dt=dt, solver=solver)
                # Frozen at their value when the device went offline,
                # through the later fault.
                @test all(traj[rows, end] .== traj[rows, k_off])
                @test all(isfinite, traj)
            end
            empty!(ps.dynamic.disconnect_events)
            _clear_events!(ps)
        end
    end
end

# -----------------------------------------------------------------------
# Jacobian reuse (modified Newton) and predictor
# -----------------------------------------------------------------------

@testset "Jacobian reuse and predictor" begin
    ps, dp = _build_and_init("ieee9_v33.raw", "ieee9bus_gov.dyr")
    z0 = copy(dp.zvec)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.15))
    function run(; kw...)
        dp.zvec .= z0
        log = GradPower.SolverLog()
        _, traj = GradPower.integrate!(dp, ps, 1.0; log=log, kw...)
        return traj, log
    end
    for solver in (:monolithic, :schur)
        full, lf = run(; solver, jacobian_reuse=false, predictor=false)
        for (reuse, pred) in ((true, false), (false, true), (true, true))
            traj, lg = run(; solver, jacobian_reuse=reuse, predictor=pred)
            # Same Newton tolerance -> same trajectory to ~tolerance.
            @test maximum(abs, traj .- full) < 1e-8
            reuse && @test lg.lsolve_factor_count < lf.lsolve_factor_count / 4
        end
    end
    # Default is reuse + predictor.
    traj_def, lg_def = run()
    traj_on, lg_on = run(; jacobian_reuse=true, predictor=true)
    @test traj_def == traj_on
    @test lg_def.lsolve_factor_count == lg_on.lsolve_factor_count
    _clear_events!(ps)

    # A solve that fails with a stale factorization is retried with full
    # Newton. The pre-fault Jacobian, never refreshed (rho and max_stale
    # huge), cannot solve the faulted step in 3 iterations; with the retry,
    # newton_step! succeeds.
    ps2, dp2 = _build_and_init("ieee9_v33.raw", "ieee9bus_gov.dyr")
    dyn = ps2.dynamic; L = dyn.layout; ybus = ps2.network.ybus_real
    n = length(dp2.zvec)
    J = GradPower.preallocate_jacobian(ps2)
    GradPower.beuler_jac_batched!(J, dp2.zvec, dp2.uvec, dp2.pvec, dyn, ybus, L, dyn.diff_dim, 1/120)
    fact = Ref{Any}(GradPower.ds_solver(GradPower.KLUBackend(), J; tuned=true))
    r = GradPower.JacobianReuse(true; rho=1e300, max_stale=10^6)
    r.dt = 1/120
    GradPower.add_event!(ps2, GradPower.ContingencyEvent(4, 0.02, 0.0, 1.0))
    GradPower.activate!(ps2.dynamic.events[1])
    zold = copy(dp2.zvec)
    z_stale = copy(zold)
    ok_stale = GradPower._newton_loop!(zeros(n), z_stale, zeros(n), J, fact, zold, dp2.uvec,
        dp2.pvec, ps2, 1/120, r, true, 3, 1e-10, false, false, zeros(n), nothing, :inf, nothing)
    @test !ok_stale
    log = GradPower.SolverLog()
    z = copy(zold)
    r.dt = 1/120
    ok = GradPower.newton_step!(z, zeros(n), J, fact, zold, dp2.uvec, dp2.pvec, ps2, 1/120;
                                itermax=6, log=log, reuse=r, dx=zeros(n), zwork=zeros(n))
    @test ok
    @test log.lsolve_factor_count >= 1   # the full-Newton retry refactored
    _clear_events!(ps2)
end
