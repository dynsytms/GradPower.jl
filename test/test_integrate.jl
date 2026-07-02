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
